"""
URL configuration for Active Learning app.
"""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('initialize/', views.initialize, name='initialize'),
    path('load-data/', views.load_data, name='load_data'),
    path('learn/', views.active_learning_loop, name='active_learning_loop'),
    path('api/llm-label-instance/', views.llm_label_instance, name='llm_label_instance'),
    path('api/llm-label-feature/', views.llm_label_feature, name='llm_label_feature'),
    path('api/llm-auto-label-batch/', views.llm_auto_label_batch, name='llm_auto_label_batch'),
    path('evaluate/', views.evaluate, name='evaluate'),
    path('predict/', views.predict, name='predict'),
]
