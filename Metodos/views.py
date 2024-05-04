from django.shortcuts import render
import numpy as np
import pandas as pd
import sympy as sp
from sympy import *

import plotly.graph_objs as go
def home(request):
    return render(request, 'All_methods.html')


def regla_falsa(request):
    if request.method == 'POST':
    
        a = float(request.POST['a'])
        b = float(request.POST['b'])
        tolerancia = float(request.POST['tolerancia'])
        funcion = request.POST['funcion']

        x = sp.symbols('x')
        fx = lambda x: eval(funcion)

        a_inicial = a
        b_inicial = b
        tolera = tolerancia

        tabla = []
        tramo = abs(b - a)
        fa = fx(a)
        fb = fx(b)

        while not(tramo <= tolera):
            c = b - fb * (a - b) / (fa - fb)
            fc = fx(c)
            tabla.append([a, c, b, fa, fc, fb, tramo])
            cambio = np.sign(fa) * np.sign(fc)
            if cambio > 0:
                tramo = abs(c - a)
                a = c
                fa = fc
            else:
                tramo = abs(b - c)
                b = c

        columnas = ['a', 'c', 'b', 'fa', 'fc', 'fb', 'Error']
        df = pd.DataFrame(tabla, columns=columnas)

        coeficientes = sp.Poly(funcion).all_coeffs()
        raices = np.real(np.roots(coeficientes))
        raices_en_intervalo = [r for r in raices if a_inicial <= r <= b_inicial]

        intervalo_x = np.arange(a_inicial, b_inicial, 0.1)
        intervalo_y = fx(intervalo_x)
        intervalo_x_completo = np.arange(a_inicial, b_inicial+0.1, 0.1)
        intervalo_y_completo = fx(intervalo_x_completo)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=intervalo_x_completo, y=intervalo_y_completo, mode='lines', name='f(x)'))
        fig.add_trace(go.Scatter(x=raices_en_intervalo, y=[0]*len(raices_en_intervalo), mode='markers', name='Raíces'))
        fig.update_layout(title=f'Función: {funcion} en intervalo [{a_inicial}, {b_inicial}]',
                          xaxis_title='x',
                          yaxis_title='f(x)')

        plot_html = fig.to_html(full_html=False, default_height=500, default_width=700)

        context = {'df': df.to_html(), 'plot_html': plot_html}
        return render(request, 'regla_falsa.html', context)

    return render(request, 'regla_falsa.html')

# bisection function, receives a tol, xi, xs, niter and function
def biseccion(request):
    if request.method == 'POST':
        mensaje=""
        xi = float(request.POST['xi'])
        xs = float(request.POST['xs'])
        tol = float(request.POST['tol'])
        niter = int(request.POST['niter'])
        funcion = request.POST['funcion']
        xi_copy=xi
        xs_copy=xs

        x = sp.symbols('x')
        funcion_expr = parse_expr(funcion, local_dict={'x': x})
        fx = lambda x: funcion_expr.subs(x, xi)

        tabla = []
        absoluto = abs(xs-xi)/2
        itera = 0
        xm = 0
        hay_solucion=False
        solucion=0
        fsolucion=0

        #fi=fx(xi)
        #fs=fx(xs)
        fi = funcion_expr.subs(x, xi).evalf()
        fs=funcion_expr.subs(x, xs).evalf()
        if fi==0:
            s=xi
            absoluto=0
            mensaje="La solución es: "+str(s)
            hay_solucion=True
            solucion=xi
            fsolucion=fi
        elif fs==0:
            s=xs
            absoluto=0
            mensaje="La solución es: "+str(s)
            hay_solucion=True
            solucion=xs
            fsolucion=fs
        elif fi*fs<0:
            xm=xi+((xs-xi)/2)
            fxm=funcion_expr.subs(x, xm).evalf()
            if(xm!=0):
                relativo=(xm-xi)/xm
            else:
                relativo=(xi)/0.0000001
            if(fxm==0):
                mensaje="La solución es: "+str(xm)
                hay_solucion=True
                solucion=xm
                fsolucion=fxm
                tabla.append([xi, xm, xs,fi, fxm, fs,0,relativo])
            else:
                tabla.append([xi, xm, xs,fi, fxm, fs,absoluto,relativo])
                while absoluto > tol and itera < niter and fxm != 0:
                    if fi*fxm<0:
                        xs=xm
                        fs=fxm
                    else:
                        xi=xm
                        fi=fxm
                    xaux=xm
                    xm=(xi+xs)/2
                    fxm=funcion_expr.subs(x, xm).evalf()
                    
                    absoluto=abs(xaux-xm)

                    if(xm!=0):
                        relativo=(xm-xaux)/xm
                    else:
                        relativo=(xaux)/0.0000001
                    itera+=1
                    
                    tabla.append([xi, xm,xs,fi, fxm, fs,absoluto,relativo])
                if fxm==0:
                    s=xm
                    mensaje="La solución es: "+str(s)
                    hay_solucion=True
                    solucion=xm
                    fsolucion=fxm
                else:
                    if absoluto<tol:
                        mensaje="La solucion aproximada que cumple la tolerancia es : "+str(xm)
                        hay_solucion=True
                        solucion=xm
                        fsolucion=fxm
                    else:
                        mensaje="Se ha alcanzado el número máximo de iteraciones"

        else:
            mensaje="No hay raiz en el intervalo, intente con otro intervalo"
        # Calculate relative error
        

        columnas = ['xi', 'xm','xs','f(xi)' ,'f(xm)','f(xs)', 'Err Abs ', 'Err Rel']
        df = pd.DataFrame(tabla, columns=columnas)
        df.index = np.arange(1, len(df) + 1)


        '''
        coeficientes = sp.Poly(funcion).all_coeffs()
        raices = np.real(np.roots(coeficientes))
        raices_en_intervalo = [r for r in raices if xi <= r <= xs]

        intervalo_x = np.arange(xi, xs, 0.1)
        intervalo_y = fx(intervalo_x)
        intervalo_x_completo = np.arange(xi, xs+0.1, 0.1)
        intervalo_y_completo = fx(intervalo_x_completo)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=intervalo_x_completo, y=intervalo_y_completo, mode='lines', name='f(x)'))
        fig.add_trace(go.Scatter(x=raices_en_intervalo, y=[0]*len(raices_en_intervalo), mode='markers', name='Raíces'))
        fig.update_layout(title=f'Función: {funcion} en intervalo [{xi}, {xs}]',
                          xaxis_title='x',
                          yaxis_title='f(x)')

        plot_html = fig.to_html(full_html=False, default_height=500, default_width=700)
        '''
        intervalo_x = np.arange(xi_copy, xs_copy,0.1)
        fx = sp.lambdify(x, funcion_expr, 'numpy')
        intervalo_y = fx(intervalo_x)
        intervalo_x_completo = np.arange(xi_copy, xs_copy,0.1)
        intervalo_y_completo = fx(intervalo_x_completo)
        
        # Crear figura
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=intervalo_x_completo, y=intervalo_y_completo, mode='lines', name='f(x)'))
        fig.add_trace(go.Scatter(x=[str(solucion)], y=[str(fsolucion)], mode='markers', name='Raíz hallada'))
        fig.update_layout(title=f'Función: {funcion} en intervalo [{xi_copy}, {xs_copy}]',
                        xaxis_title='x',
                        yaxis_title='f(x)')
        
        plot_html = fig.to_html(full_html=False, default_height=500, default_width=700)
        context = {'df': df.to_html(), 'plot_html': plot_html}
        context['mensaje']=mensaje
        context['nombre_metodo']="Bisección"
        return render(request, 'one_method.html', context)

def secante(request):
    if request.method == 'POST':
        mensaje=""
        x0 = float(request.POST['x0'])
        x1 = float(request.POST['x1'])
        tol = float(request.POST['tol'])
        niter = int(request.POST['niter'])
        funcion = request.POST['funcion']

        x = sp.symbols('x')
        fx = lambda x: eval(funcion)

        print(f'x0: {x0}, x1: {x1}, tol: {tol}, niter: {niter}, funcion: {funcion}')

        tabla = []
        itera = 1
        x2 = 0
        f0=fx(x0)
        f1=fx(x1)
        if f0==0:
            s=x0
            mensaje="La solución es: "+str(s)
        elif f1==0:
            s=x1
            mensaje="La solución es: "+str(s)
        else:
            x2=x1-((f1*(x0-x1))/(f0-f1))
            f2=fx(x2)
            if x2!=0:
                relativo=abs((x2-x1)/x2)*100
            else:
                relativo=(abs(x1)/0.0000001)*100
            if f2==0:
                mensaje="La solución es: "+str(x2)
                tabla.append([itera,x0, x1, x2, f0, f1, f2, 0,"-","-"])
            else:
                tabla.append([itera,x0, x1, x2, f0, f1, f2, abs(x2-x1),"-","-"])
                x0=x1
                f0=f1
                x1=x2
                f1=f2
                itera+=1
                while itera < niter and tol < relativo and f1!=0 and f0!=0:
                    x2 = x1-((f1*(x0-x1))/(f0-f1))
                    f2 = fx(x2)
                    if x2!=0:
                        relativo=abs((x2-x1)/x2)*100
                    else:
                        relativo=(abs(x1)/0.0000001)*100
                    if f2==0:
                        mensaje="La solución es: "+str(x2)+"\n En la iteracion "+str(itera)
                        tabla.append([itera,x0, x1, x2, f0, f1, f2, 0,relativo,relativo*100])
                        break     
                    absoluto=(x2-x1)
                    tabla.append([itera,x0, x1, x2, f0, f1, f2, absoluto,relativo,relativo*100])
                    x0=x1
                    f0=f1
                    x1=x2
                    f1=f2
                    itera+=1
                if f1==0:
                    mensaje="La solución es: "+str(x1)
                if relativo<tol:
                    mensaje="La solución aproximada que cumple la tolerancia es : "+str(x2)+" En la iteracion "+str(itera)+" = x"+str(itera)
                elif itera==niter:
                    mensaje="Se ha alcanzado el número máximo de iteraciones"

        columnas = ['i','xi-1', 'xi', 'xi+1', 'f(xi-1)', 'f(xi)', 'f(xi+1)','Err Abs', 'Err Rel','Err Rel %']
        df = pd.DataFrame(tabla, columns=columnas)

        intervalo_x = np.arange(x0, x1, 0.1)
        intervalo_y = fx(intervalo_x)
        intervalo_x_completo = np.arange(x0, x1+0.1, 0.1)
        intervalo_y_completo = fx(intervalo_x_completo)
        
        # Crear figura
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=intervalo_x_completo, y=intervalo_y_completo, mode='lines', name='f(x)'))
        fig.update_layout(title=f'Función: {funcion} en intervalo [{x0}, {x1}]',
                        xaxis_title='x',
                        yaxis_title='f(x)')
        
        plot_html = fig.to_html(full_html=False, default_height=500, default_width=700)
        context = {'df': df.to_html(), 'plot_html': plot_html}
        context['mensaje']=mensaje
        context['nombre_metodo']="Secante"
        return render(request, 'one_method.html', context)

        