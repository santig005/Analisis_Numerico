from django.shortcuts import render
import numpy as np
import pandas as pd
import sympy as sp
from sympy import *
from django.utils.safestring import mark_safe
import plotly.graph_objects as go
from .metodos.iterativos import metodo_gauss_seidel,metodo_jacobi
from .utiles.saver import dataframe_to_txt

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
        xi = (request.POST['xi'])
        xs = (request.POST['xs'])
        tol = (request.POST['tol'])
        niter = (request.POST['niter'])
        funcion = request.POST['funcion']
        tipo_error = request.POST.get('tipo_error')
        if xi is None or xs is None or tol is None or niter is None or funcion is None or tipo_error is None:
            xi = 1
            xs = 2
            tol = 0.0001
            niter = 100
            funcion = 'x**2 - 3'
            tipo_error = 'absoluto'
            mensaje+="Se tomo el caso por defecto, por favor ingrese los valores de todos los campos <br> "
        else:
            xi=float(xi)
            xs=float(xs)
            tol=float(tol)
            niter=int(niter)
        xi_copy=xi
        xs_copy=xs
        hay_solucion=False
        solucion=0
        fsolucion=0
        try:
            x = sp.symbols('x')
            funcion_expr = parse_expr(funcion, local_dict={'x': x})
            fx = lambda x: funcion_expr.subs(x, xi)
            tabla = []
            absoluto = abs(xs-xi)/2
            itera = 0
            xm = 0
            fi = funcion_expr.subs(x, xi).evalf()
            fs=funcion_expr.subs(x, xs).evalf()
            if fi==0:
                s=xi
                absoluto=0
                mensaje+="La solución es: "+str(s)
                hay_solucion=True
                solucion=xi
                fsolucion=fi
            elif fs==0:
                s=xs
                absoluto=0
                mensaje+="La solución es: "+str(s)
                hay_solucion=True
                solucion=xs
                fsolucion=fs
            elif fi*fs<0:
                xm=xi+((xs-xi)/2)
                fxm=funcion_expr.subs(x, xm).evalf()
                if(xm!=0):
                    relativo=abs((xm-xi)/xm)
                else:
                    relativo=abs((xi)/0.0000001)
                if(fxm==0):
                    mensaje+="La solución es: "+str(xm)
                    hay_solucion=True
                    solucion=xm
                    fsolucion=fxm
                    tabla.append([xi, xm, xs,fi, fxm, fs,0,relativo])
                else:
                    tabla.append([xi, xm, xs,fi, fxm, fs,absoluto,relativo])
                    if tipo_error=="absoluto":
                        error=absoluto
                    else:
                        error=relativo
                    while error > tol and itera < niter and fxm != 0:
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
                            relativo=abs((xm-xaux)/xm)
                        else:
                            relativo=abs((xaux)/0.0000001)
                        itera+=1
                        if tipo_error=="absoluto":
                            error=absoluto
                        else:
                            error=relativo
                        tabla.append([xi, xm,xs,fi, fxm, fs,absoluto,relativo])
                    if fxm==0:
                        s=xm
                        mensaje+="La solución es: "+str(s)
                        hay_solucion=True
                        solucion=xm
                        fsolucion=fxm
                    else:
                        if error<tol:
                            mensaje+="La solucion aproximada que cumple la tolerancia es : "+str(xm)
                            hay_solucion=True
                            solucion=xm
                            fsolucion=fxm
                        else:
                            mensaje+="Se ha alcanzado el número máximo de iteraciones"

            else:
                mensaje+="No hay raiz en el intervalo, intente con otro intervalo"
            

            columnas = ['xi', 'xm','xs','f(xi)' ,'f(xm)','f(xs)', 'Err abs ', 'Err Rel']
            df = pd.DataFrame(tabla, columns=columnas)
            df.index = np.arange(1, len(df) + 1)
            
            
            intervalo_x = np.arange(xi_copy, xs_copy,0.1)
            fx = sp.lambdify(x, funcion_expr, 'numpy')
            intervalo_y = fx(intervalo_x)
            intervalo_x_completo = np.arange(xi_copy, xs_copy,0.1)
            intervalo_y_completo = fx(intervalo_x_completo)
            
            # Crear figura


            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=intervalo_x_completo, y=intervalo_y_completo, mode='lines', name='f(x)'))
            if hay_solucion:
                fig.add_trace(go.Scatter(x=[str(solucion)], y=[str(fsolucion)], mode='markers', name='Raíz hallada'))
            fig.update_layout(title=f'Función: {funcion} en intervalo [{xi_copy}, {xs_copy}]',
                            xaxis_title='x',
                            yaxis_title='f(x)')
            
            plot_html = fig.to_html(full_html=False, default_height=500, default_width=700)
            dataframe_to_txt(df,"biseccion")
            context = {'df': df.to_html(), 'plot_html': plot_html}
            mensaje=mark_safe(mensaje)
            context['mensaje']=mensaje
            context['nombre_metodo']="Bisección"
            return render(request, 'one_method.html', context)
        except:
            mensaje+="Error en los datos ingresados, por favor verifique que los datos sean correctos"
            context = {'mensaje':mensaje}
            return render(request, 'one_method.html', context)

def secante(request):
    if request.method == 'POST':
        mensaje=""
        x0 = (request.POST['x0'])
        x1 = (request.POST['x1'])
        tol = (request.POST['tol'])
        niter = (request.POST['niter'])
        funcion = request.POST['funcion']
        tipo_error = request.POST.get('tipo_error')
        if x0 is None or x1 is None or tol is None or niter is None or funcion is None or tipo_error is None:
            x0 = 1
            x1 = 2
            tol = 0.0001
            niter = 100
            funcion = 'x**2 - 3'
            tipo_error = 'absoluto'
            mensaje+="Se tomo el caso por defecto, por favor ingrese los valores de todos los campos <br> "
        else:
            x0=float(x0)
            x1=float(x1)
            tol=float(tol)
            niter=int(niter)

        x = sp.symbols('x')
        funcion_expr = parse_expr(funcion, local_dict={'x': x})
        fx = lambda x: funcion_expr.subs(x, x0)

        xi_copy=x0
        xs_copy=x1
        hay_solucion=False
        solucion=0
        fsolucion=0
        try:
            tabla = []
            itera = 1
            x2 = 0
            f0=funcion_expr.subs(x, x0).evalf()
            f1=funcion_expr.subs(x, x1).evalf()
            if f0==0:
                s=x0
                mensaje+="La solución es: "+str(s)
                hay_solucion=True
                solucion=x0
                fsolucion=f0
            elif f1==0:
                s=x1
                mensaje+="La solución es: "+str(s)
                hay_solucion=True
                solucion=x1
                fsolucion=f1
            else:
                x2=x1-((f1*(x0-x1))/(f0-f1))
                f2=funcion_expr.subs(x, x2).evalf()
                if x2!=0:
                    relativo=abs((x2-x1)/x2)
                else:
                    relativo=(abs(x1)/0.0000001)
                absoluto=abs(x2-x1)
                if f2==0:
                    mensaje+="La solución es: "+str(x2)
                    hay_solucion=True
                    solucion=x2
                    fsolucion=f2
                    tabla.append([itera,x0, x1, x2, f0, f1, f2, 0,"-"])
                else:
                    tabla.append([itera,x0, x1, x2, f0, f1, f2, abs(x2-x1),"-"])
                    x0=x1
                    f0=f1
                    x1=x2
                    f1=f2
                    itera+=1
                    if tipo_error=="absoluto":
                        error=absoluto
                    else:
                        error=relativo
                    while itera < niter and tol < error and f1!=0 and f0!=0:
                        x2 = x1-((f1*(x0-x1))/(f0-f1))
                        f2 = funcion_expr.subs(x, x2).evalf()
                        if x2!=0:
                            relativo=abs((x2-x1)/x2)*100
                        else:
                            relativo=(abs(x1)/0.0000001)*100
                        if f2==0:
                            mensaje+="La solución es: "+str(x2)+"\n En la iteracion "+str(itera)
                            hay_solucion=False
                            solucin=x2
                            fsolucion=f2
                            tabla.append([itera,x0, x1, x2, f0, f1, f2, 0,relativo])
                            break     
                        absoluto=abs(x2-x1)
                        tabla.append([itera,x0, x1, x2, f0, f1, f2, absoluto,relativo])
                        x0=x1
                        f0=f1
                        x1=x2
                        f1=f2
                        itera+=1
                        if tipo_error=="absoluto":
                            error=absoluto
                        else:
                            error=relativo
                    if f1==0:
                        mensaje+="La solución es: "+str(x1)
                        hay_solucion=True
                        solucion=x1
                        fsolucion=f1
                    if error<tol:
                        mensaje+="La solución aproximada que cumple la tolerancia es : "+str(x2)+" En la iteracion "+str(itera)+" = x"+str(itera)
                        hay_solucion=True
                        solucion=x2
                        fsolucion=f2
                    elif itera==niter:
                        mensaje+="Se ha alcanzado el número máximo de iteraciones"

            columnas = ['i','xi-1', 'xi', 'xi+1', 'f(xi-1)', 'f(xi)', 'f(xi+1)','Err abs', 'Err Rel']
            df = pd.DataFrame(tabla, columns=columnas)
            intervalo_x = np.arange(xi_copy, xs_copy,0.1)
            fx = sp.lambdify(x, funcion_expr, 'numpy')
            intervalo_y = fx(intervalo_x)
            intervalo_x_completo = np.arange(xi_copy, xs_copy,0.1)
            intervalo_y_completo = fx(intervalo_x_completo)
            
            # Crear figura
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=intervalo_x_completo, y=intervalo_y_completo, mode='lines', name='f(x)'))
            if hay_solucion:
                fig.add_trace(go.Scatter(x=[str(solucion)], y=[str(fsolucion)], mode='markers', name='Raíz hallada'))
            fig.update_layout(title=f'Función: {funcion} en intervalo [{xi_copy}, {xs_copy}]',
                            xaxis_title='x',
                            yaxis_title='f(x)')
            
            plot_html = fig.to_html(full_html=False, default_height=500, default_width=700)
            dataframe_to_txt(df,"secante")
            context = {'df': df.to_html(), 'plot_html': plot_html}
            mensaje=mark_safe(mensaje)
            context['mensaje']=mensaje
            context['nombre_metodo']="Secante"
            return render(request, 'one_method.html', context)
        except:
            mensaje+="Error en los datos ingresados, por favor verifique que los datos sean correctos"
            mensaje=mark_safe(mensaje)
            context = {'mensaje':mensaje}
            return render(request, 'one_method.html', context)

def iterativos(request):
    if request.method == 'POST':
        tamaño=request.POST['numero']
        metodo=request.POST['metodo_iterativo']
        tol=request.POST['tol']
        niter=request.POST['niter']
        try:
            matriz=[]
            tamaño=int(tamaño)
            tol=float(tol)
            niter=int(niter)
            i=0
            for i in range(tamaño):
                row = []
                j=0
                for j in range(tamaño):
                    val = request.POST.get(f'matrix_cell_{i}_{j}')
                    row.append(int(val) if val else 0)
                matriz.append(row)
            vectorx=[]
            for i in range(tamaño):
                val = request.POST.get(f'vx_cell_{i}')
                vectorx.append(int(val) if val else 0)
            vectorb=[]
            for i in range(tamaño):
                val = request.POST.get(f'vb_cell_{i}')
                vectorb.append(int(val) if val else 0)
            nmatriz=np.array(matriz)
            nvectorx = np.array(vectorx).reshape(-1, 1)
            nvectorb = np.array(vectorb).reshape(-1, 1)
            print("vamos para el if")
            if metodo=="jacobi":
                context=metodo_jacobi(nmatriz,nvectorx,nvectorb,tol,niter)
            elif metodo=="gauss_seidel":
                context=metodo_gauss_seidel(nmatriz,nvectorx,nvectorb,tol,niter)
            elif metodo=="sor":
                context=metodo_sor(nmatriz,nvectorx,nvectorb,tol,niter)
            return render(request,'resultado_iterativo.html',context)

        except:
            context={'mensaje':'No se pudo realizar la operación'}
            return render(request,'resultado_iterativo.html',context)



        