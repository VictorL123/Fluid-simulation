from django.urls import path
from . import views

urlpatterns = [
    path('jobs/',              views.jobs,        name='jobs'),
    path('jobs/<int:job_id>/', views.job_detail,  name='job_detail'),
    path('jobs/<int:job_id>/stream/', views.job_stream, name='job_stream'),
    path('status/',            views.api_status,  name='api_status'),
]
