from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('regla_falsa/', views.reglaFalsaView, name='regla_falsa'),
    path('biseccion/', views.biseccion, name='biseccion'),
    path('PuntoFijo/', views.puntoFijoView, name='PuntoFijo'),
    path('secante/', views.secante, name='secante'),
    path('iterativos/',views.iterativos,name="iterativos"),
    path('Interpolacion/',views.interpolacion,name="interpolacion"),
]
