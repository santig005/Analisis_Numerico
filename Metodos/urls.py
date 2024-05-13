from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('regla_falsa/', views.regla_falsa, name='regla_falsa'),
    path('biseccion/', views.biseccion, name='biseccion'),
    path('secante/', views.secante, name='secante'),
    path('iterativos/',views.iterativos,name="iterativos")
]
