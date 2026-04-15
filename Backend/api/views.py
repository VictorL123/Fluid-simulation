"""
api/views.py
=============
REST API endpoints for the fluid simulation platform.

Endpoints:
    POST /api/jobs/          — submit a new simulation job
    GET  /api/jobs/          — list all jobs
    GET  /api/jobs/<id>/     — get job details + results
    GET  /api/jobs/<id>/stream/ — SSE stream of live progress
    POST /api/jobs/<id>/surrogate/ — run surrogate on a job's nu value
    GET  /api/status/        — API health check
"""

import os
import sys
import glob
import json
import time
import subprocess
import threading
import numpy as np
from datetime import datetime, timezone

from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone as tz
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .models import SimulationJob
from .surrogate import run_surrogate
from django.conf import settings


# ============================================================
#  Helpers
# ============================================================
def get_solver_binary():
    """Find the C++ solver binary on Windows or Linux."""
    candidates = [
        settings.SOLVER_BIN_WINDOWS,
        settings.SOLVER_BIN_UNIX,
        os.path.join(settings.SOLVER_DIR, 'fluid_sim.exe'),
        os.path.join(settings.SOLVER_DIR, 'fluid_sim'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def job_to_dict(job, include_fields=False):
    """Serialise a SimulationJob to a dictionary."""
    d = {
        'id':           job.id,
        'nu':           job.nu,
        're':           job.re,
        'total_steps':  job.total_steps,
        'grid_size':    job.grid_size,
        'status':       job.status,
        'source':       job.source,
        'progress':     job.progress,
        'created_at':   job.created_at.isoformat() if job.created_at else None,
        'finished_at':  job.finished_at.isoformat() if job.finished_at else None,
        'duration_s':   job.duration_seconds(),
        'error_msg':    job.error_msg,
    }
    if include_fields and job.status == 'complete':
        u, v, p = job.get_fields()
        if u is not None:
            N = job.grid_size
            d['fields'] = {
                'u': u.tolist(),
                'v': v.tolist(),
                'p': p.tolist(),
                'N': N,
            }
    return d


# ============================================================
#  Job list and submission
# ============================================================
@csrf_exempt
@api_view(['GET', 'POST'])
def jobs(request):
    """
    GET  — return list of all jobs (newest first)
    POST — submit a new simulation job
    """
    if request.method == 'GET':
        all_jobs = SimulationJob.objects.all()[:20]
        return Response([job_to_dict(j) for j in all_jobs])

    if request.method == 'POST':
        data = request.data

        # Validate inputs
        try:
            nu = float(data.get('nu', 0.01))
            total_steps = int(data.get('total_steps', 5000))
            grid_size = int(data.get('grid_size', 41))
            source = data.get('source', 'solver')
        except (ValueError, TypeError) as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        if not (0.001 <= nu <= 0.5):
            return Response({'error': 'nu must be between 0.001 and 0.5'},
                            status=status.HTTP_400_BAD_REQUEST)
        if not (100 <= total_steps <= 50000):
            return Response({'error': 'total_steps must be between 100 and 50000'},
                            status=status.HTTP_400_BAD_REQUEST)
        if grid_size not in [41, 65, 81, 129]:
            return Response({'error': 'grid_size must be 41, 65, 81, or 129'},
                            status=status.HTTP_400_BAD_REQUEST)
        if source not in ['solver', 'surrogate']:
            return Response({'error': 'source must be solver or surrogate'},
                            status=status.HTTP_400_BAD_REQUEST)

        # Create job
        job = SimulationJob.objects.create(
            nu=nu,
            re=round(1.0 / nu, 2),
            total_steps=total_steps,
            grid_size=grid_size,
            source=source,
            status='queued',
        )

        # Run in background thread so API returns immediately
        if source == 'solver':
            thread = threading.Thread(target=run_solver_job, args=(job.id,))
        else:
            thread = threading.Thread(target=run_surrogate_job, args=(job.id,))
        thread.daemon = True
        thread.start()

        return Response(job_to_dict(job), status=status.HTTP_201_CREATED)


# ============================================================
#  Job detail
# ============================================================
@api_view(['GET'])
def job_detail(request, job_id):
    """Return full job details including result fields if complete."""
    try:
        job = SimulationJob.objects.get(id=job_id)
    except SimulationJob.DoesNotExist:
        return Response({'error': 'Job not found'}, status=status.HTTP_404_NOT_FOUND)

    return Response(job_to_dict(job, include_fields=True))


# ============================================================
#  SSE live progress stream
# ============================================================
def job_stream(request, job_id):
    """
    Server-Sent Events endpoint — streams live progress updates
    to the frontend while a job is running.

    The frontend connects to this endpoint and receives events
    every second until the job completes or fails.
    """
    def event_generator():
        while True:
            try:
                job = SimulationJob.objects.get(id=job_id)
            except SimulationJob.DoesNotExist:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break

            payload = {
                'status':   job.status,
                'progress': job.progress,
                'total':    job.total_steps,
                'pct':      round(job.progress / max(job.total_steps, 1) * 100, 1),
            }
            yield f"data: {json.dumps(payload)}\n\n"

            if job.status in ('complete', 'failed'):
                break

            time.sleep(1.0)

    response = StreamingHttpResponse(
        event_generator(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response


# ============================================================
#  API status
# ============================================================
@api_view(['GET'])
def api_status(request):
    """Health check — also reports whether solver binary exists."""
    binary = get_solver_binary()
    return Response({
        'status':          'ok',
        'solver_available': binary is not None,
        'solver_path':      binary,
        'total_jobs':       SimulationJob.objects.count(),
    })


# ============================================================
#  Background job runners
# ============================================================
def run_solver_job(job_id):
    """
    Runs the C++ solver as a subprocess.
    Updates job.progress periodically by monitoring output files.
    Saves final fields to the database when complete.
    """
    import pandas as pd

    job = SimulationJob.objects.get(id=job_id)
    job.status = 'running'
    job.started_at = tz.now()
    job.save()

    binary = get_solver_binary()
    if binary is None:
        job.status = 'failed'
        job.error_msg = 'Solver binary not found. Run make omp in solver/'
        job.save()
        return

    solver_dir = settings.SOLVER_DIR
    output_dir = os.path.join(solver_dir, 'output')

    # Clear old output files
    for f in glob.glob(os.path.join(output_dir, 'step_*.csv')):
        os.remove(f)

    # Run solver — passes nu, steps, output_every as arguments
    output_every = max(job.total_steps // 20, 100)  # ~20 progress updates
    process = subprocess.Popen(
        [binary, str(job.nu), str(job.total_steps), str(output_every)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=solver_dir
    )

    # Poll progress while solver runs
    while process.poll() is None:
        # Count output files to estimate progress
        files = sorted([
            f for f in glob.glob(os.path.join(output_dir, 'step_*.csv'))
            if not f.endswith('step_00000.csv')
        ])
        if files:
            # Extract step number from filename
            last_file = os.path.basename(files[-1])
            try:
                step = int(last_file.replace('step_', '').replace('.csv', ''))
                job.progress = step
                job.save()
            except ValueError:
                pass
        time.sleep(1.0)

    # Check if solver succeeded
    if process.returncode != 0:
        stderr = process.stderr.read().decode()
        job.status = 'failed'
        job.error_msg = stderr[:500]
        job.finished_at = tz.now()
        job.save()
        return

    # Read final output file
    output_files = sorted([
        f for f in glob.glob(os.path.join(output_dir, 'step_*.csv'))
        if not f.endswith('step_00000.csv')
    ])
    if not output_files:
        job.status = 'failed'
        job.error_msg = 'No output files found after solver completed'
        job.finished_at = tz.now()
        job.save()
        return

    try:
        df = pd.read_csv(output_files[-1])
        N  = int(df['i'].max()) + 1
        u  = df['u'].values.reshape(N, N).astype(np.float32)
        v  = df['v'].values.reshape(N, N).astype(np.float32)
        p  = df['p'].values.reshape(N, N).astype(np.float32)

        job.set_fields(u, v, p)
        job.status = 'complete'
        job.progress = job.total_steps
        job.source = 'solver'
        job.finished_at = tz.now()
        job.save()
    except Exception as e:
        job.status = 'failed'
        job.error_msg = str(e)
        job.finished_at = tz.now()
        job.save()


def run_surrogate_job(job_id):
    """
    Runs the surrogate model (mock or real) for a job.
    Much faster than the C++ solver.
    """
    job = SimulationJob.objects.get(id=job_id)
    job.status = 'running'
    job.started_at = tz.now()
    job.save()

    try:
        (u, v, p), is_real = run_surrogate(job.nu, job.grid_size)
        job.set_fields(u, v, p)
        job.status = 'complete'
        job.progress = job.total_steps
        job.source = 'surrogate' if is_real else 'solver'
        job.finished_at = tz.now()
        job.save()
    except Exception as e:
        job.status = 'failed'
        job.error_msg = str(e)
        job.finished_at = tz.now()
        job.save()
