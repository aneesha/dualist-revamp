"""
URL configuration for Dualist Active Learning UI.
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('active_learning.urls')),
]
