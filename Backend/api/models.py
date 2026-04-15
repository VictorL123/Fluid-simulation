from django.db import models
import json

class SimulationJob(models.Model):
    """
    Represents one simulation run submitted by the user.
    Stores input parameters and output results.
    """

    STATUS_CHOICES = [
        ('queued',    'Queued'),
        ('running',   'Running'),
        ('complete',  'Complete'),
        ('failed',    'Failed'),
    ]

    SOURCE_CHOICES = [
        ('solver',    'C++ Solver'),
        ('surrogate', 'Neural Surrogate'),
    ]

    # Input parameters
    nu          = models.FloatField(help_text="Kinematic viscosity")
    re          = models.FloatField(help_text="Reynolds number (1/nu)")
    total_steps = models.IntegerField(default=5000)
    grid_size   = models.IntegerField(default=41)

    # Job metadata
    status      = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')
    source      = models.CharField(max_length=20, choices=SOURCE_CHOICES, default='solver')
    created_at  = models.DateTimeField(auto_now_add=True)
    started_at  = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    progress    = models.IntegerField(default=0, help_text="Steps completed")
    error_msg   = models.TextField(blank=True)

    # Results — stored as JSON strings
    # Each is a flattened NxN array
    u_field     = models.TextField(blank=True, help_text="u-velocity field JSON")
    v_field     = models.TextField(blank=True, help_text="v-velocity field JSON")
    p_field     = models.TextField(blank=True, help_text="pressure field JSON")

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Job {self.id} | Re={self.re:.0f} | {self.status}"

    def set_fields(self, u, v, p):
        """Save numpy arrays as JSON strings."""
        self.u_field = json.dumps(u.tolist())
        self.v_field = json.dumps(v.tolist())
        self.p_field = json.dumps(p.tolist())

    def get_fields(self):
        """Load numpy arrays from JSON strings."""
        import numpy as np
        if not self.u_field:
            return None, None, None
        return (
            np.array(json.loads(self.u_field)),
            np.array(json.loads(self.v_field)),
            np.array(json.loads(self.p_field)),
        )

    def duration_seconds(self):
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None
