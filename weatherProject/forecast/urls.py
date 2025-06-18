from django.urls import path
from . import views  # or from forecast import views

urlpatterns = [
    path('', views.home, name='home'),  # example view
]
