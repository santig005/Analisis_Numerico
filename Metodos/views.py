from django.shortcuts import render
import numpy as np
import pandas as pd
import sympy as sp
from sympy import *
from django.utils.safestring import mark_safe
import plotly.graph_objects as go
from .metodos.iterativos import metodo_gauss_seidel,metodo_jacobi,metodo_sor
from .utiles.saver import dataframe_to_txt,plot_to_png,text_to_txt
from .utiles.plotter import plot_fx_puntos,fx_plot,spline_plot

def home(request):
    return render(request, 'All_methods.html')

def reglaFalsaView(request):
    context = {}
    
    if request.method == 'POST':
        try: 
            fx = request.POST["funcion"]

            x0 = request.POST["a"]
            X0 = float(x0)

            xi = request.POST["b"]
            Xi = float(xi)

            tol = request.POST["tolerancia"]
            Tol = float(tol)

            niter = request.POST["iteraciones"]
            Niter = int(niter)

            tipo_error = request.POST.get('tipo_error')
            
            datos = reglaFalsa(X0, Xi, Niter, Tol, fx, tipo_error)

            if "errors" in datos and datos["errors"]:
                context['error_message'] = f'Hubo un error en el metodo Regla Falsa en: {datos["errors"]}'
                return render(request, 'error.html', context)
            
            if "results" in datos:
                df = pd.DataFrame(datos["results"], columns=datos["columns"])
                
                x = sp.symbols('x')
                funcion_expr = sp.sympify(fx)

                xi_copy = X0
                xs_copy = Xi

                intervalo_x = np.arange(xi_copy, xs_copy, 0.1)
                fx_func = sp.lambdify(x, funcion_expr, 'numpy')
                intervalo_y = fx_func(intervalo_x)
                intervalo_x_completo = np.arange(xi_copy, xs_copy, 0.1)
                intervalo_y_completo = fx_func(intervalo_x_completo)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=intervalo_x_completo, y=intervalo_y_completo, mode='lines', name='f(x)'))

                if datos.get("root") is not None:
                    fig.add_trace(go.Scatter(x=[float(datos["root"])], y=[float(0)], mode='markers', name='Raíz hallada'))

                fig.update_layout(title=f'Función: {fx} en intervalo [{xi_copy}, {xs_copy}]',
                                    xaxis_title='x',
                                    yaxis_title='f(x)')

                plot_html = fig.to_html(full_html=False, default_height=500, default_width=700)
                dataframe_to_txt(df, f'Regla Falsa{fx}')
                plot_to_png(fig,f'Regla Falsa {fx}')
                
                context = {'df': df.to_html(), 'plot_html': plot_html, 'mensaje': f'La solución es: {datos["root"]}'}
        except Exception as e:
            context = {'error_message': f'Hubo un error en el metodo Regla Falsa en: {str(e)}'}
            return render(request, 'error.html', context)

    return render(request, 'one_method.html', context)

def reglaFalsa(a, b, Niter, Tol, fx, tipo_error):

    output = {
        "columns": ["iter", "a", "xm", "b", "f(xm)", "E"],
        "results": [],
        "errors": [],
        'root': '0'
    }

    #configuración inicial
    datos = list()
    x = sp.Symbol('x')
    i = 1
    cond = Tol
    error = 1.0000000

    Fun = sympify(fx)
   
    xm = 0
    xm0 = 0
    Fx_2 = 0
    Fx_3 = 0
    Fa = 0
    Fb = 0

    try:
        if Fun.subs(x, a)*Fun.subs(x, b) > 0:
            output["errors"].append("La funcion no cambia de signo en el intervalo dado")
            return output
        while (error > cond) and (i < Niter):
            if i == 1:
                Fx_2 = Fun.subs(x, a)
                Fx_2 = Fx_2.evalf()
                Fa = Fx_2

                Fx_2 = Fun.subs(x, b)
                Fx_2 = Fx_2.evalf()
                Fb = Fx_2

                xm = (Fb*a - Fa*b)/(Fb-Fa)
                Fx_3 = Fun.subs(x, xm)
                Fx_3 = Fx_3.evalf()
                datos.append([i, '{:^15.7f}'.format(a), '{:^15.7f}'.format(xm), '{:^15.7f}'.format(b), '{:^15.7E}'.format(Fx_3)])
            else:

                if (Fa*Fx_3 < 0):
                    b = xm
                else:
                    a = xm

                xm0 = xm
                Fx_2 = Fun.subs(x, a) #Función evaluada en a
                Fx_2 = Fx_2.evalf()
                Fa = Fx_2

                Fx_2 = Fun.subs(x, b) #Función evaluada en a
                Fx_2 = Fx_2.evalf()
                Fb = Fx_2

                xm = (Fb*a - Fa*b)/(Fb-Fa) #Calcular intersección en la recta en el eje x

                Fx_3 = Fun.subs(x, xm) #Función evaluada en xm (f(xm))
                Fx_3 = Fx_3.evalf()

                if tipo_error == "absoluto":
                    error = Abs(xm-xm0)
                    er = sympify(error)
                    error = er.evalf()
                else:
                    error = Abs(xm-xm0)/xm
                    er = sympify(error)
                    error = er.evalf()

                datos.append([i, '{:^15.7f}'.format(a), '{:^15.7f}'.format(xm), '{:^15.7f}'.format(b), '{:^15.7E}'.format(Fx_3), '{:^15.7E}'.format(error)])
            i += 1
    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append(
                "Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))

        return output

    output["results"] = datos
    output["root"] = xm
    return output


def puntoFijo(X0, Tol, Niter, fx, gx, tipo_error):

    output = {
        "columns": ["iter", "xi", "g(xi)", "f(xi)", "E"],
        "results": [],
        "errors": []
    }

    # Configuración inicial
    x = sp.Symbol('x')
    i = 1
    error = 1.000

    Fx = sp.sympify(fx)
    Gx = sp.sympify(gx)

    # Iteración 0
    xP = X0  # Valor inicial (Punto evaluación)
    xA = 0.0

    Fa = Fx.subs(x, xP)  # Función evaluada en el valor inicial
    Fa = Fa.evalf()

    Ga = Gx.subs(x, xP)  # Función G evaluada en el valor inicial
    Ga = Ga.evalf()

    datos = [[0, float(xP), float(Ga), float(Fa), None]]

    try:
        while error > Tol and i < Niter:  # Se repite hasta que el error sea menor a la tolerancia

            # Se evalúa el valor inicial en G, para posteriormente evaluar este valor en la función F siendo-> Xn=G(x) y F(xn) = F(G(x))
            Ga = Gx.subs(x, xP)  # Función G evaluada en el punto de inicial
            xA = Ga.evalf()

            Fa = Fx.subs(x, xA)  # Función evaluada en el valor de la evaluación de G
            Fa = Fa.evalf()

            if tipo_error == "absoluto":
                error = abs(xA - xP)
            else:
                error = abs(xA - xP) / abs(xA)
                

            datos.append([i, float(xA), float(Ga), float(Fa), float(error)])

            xP = xA  # Nuevo punto de evaluación (Punto inicial)
            i += 1

    except BaseException as e:
        output["errors"].append("Error in data: " + str(e))
        return output

    output["results"] = datos
    output["root"] = xA
    return output

def puntoFijoView(request):
    datos = ()
    if request.method == 'POST':
        try:
            fx = request.POST["funcion-F"]
            gx = request.POST["funcion-G"]

            x0 = request.POST["vInicial"]
            X0 = float(x0)

            tol = request.POST["tolerancia"]
            Tol = float(tol)

            niter = request.POST["iteraciones"]
            Niter = int(niter)

            tipo_error = request.POST.get('tipo_error')

            datos = puntoFijo(X0,Tol,Niter,fx,gx, tipo_error)
            df = pd.DataFrame(datos["results"], columns=datos["columns"])

            x = sp.symbols('x')
            funcion_f = sp.sympify(fx)
            funcion_g = sp.sympify(gx)

            intervalo_x = np.linspace(X0 - 5, X0 + 5, 400)  # Ajustar el intervalo si es necesario
            fx_func = sp.lambdify(x, funcion_f, 'numpy')
            gx_func = sp.lambdify(x, funcion_g, 'numpy')
            intervalo_y_f = fx_func(intervalo_x)
            intervalo_y_g = gx_func(intervalo_x)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=intervalo_x, y=intervalo_y_f, mode='lines', name='f(x)'))
            fig.add_trace(go.Scatter(x=intervalo_x, y=intervalo_y_g, mode='lines', name='g(x)'))
            
            if datos.get("root") is not None:
                fig.add_trace(go.Scatter(x=[float(datos["root"])], y=[0], mode='markers', name='Raíz hallada'))

            fig.update_layout(title=f'Función: {fx} y {gx} en intervalo [{X0 - 5}, {X0 + 5}]',
                            xaxis_title='x',
                            yaxis_title='f(x) / g(x)')
            
            plot_html = fig.to_html(full_html=False, default_height=500, default_width=700)

            dataframe_to_txt(df, f'PuntoFijo {fx}')
            plot_to_png(fig,f'PuntoFijo {fx}')
            
        
            if datos:
                context = {'df': df.to_html(), 'plot_html': plot_html, 'mensaje': f'La solucion es: {datos["root"]}'}
                return render(request, 'one_method.html', context)

            return render(request, 'one_method.html')
        
        except Exception as e:
            context = {'error_message': f'Hubo un error en el metodo Punto Fijo en: {str(e)}'}
            return render(request, 'error.html', context)

    return render(request, 'one_method.html')
            

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
            
            if hay_solucion:
                fig=plot_fx_puntos(funcion,xi_copy,xs_copy,[str(solucion)],[str(fsolucion)],'raíz hallada')
            else:
                fig=fx_plot(funcion,xi_copy,xs_copy)
            plot_html = fig.to_html(full_html=False, default_height=500, default_width=700)
            dataframe_to_txt(df,"biseccion")
            plot_to_png(fig,"biseccion")
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
            if hay_solucion:
                fig=plot_fx_puntos(funcion,xi_copy,xs_copy,[str(solucion)],[str(fsolucion)],'raíz hallada')
            else:
                fig=fx_plot(funcion,xi_copy,xs_copy)
            
            plot_html = fig.to_html(full_html=False, default_height=500, default_width=700)
            dataframe_to_txt(df,"secante")
            plot_to_png(fig,"secante")
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
        

def newton(x0, Tol, Niter, fx, df, tipo_error):

    output = {
        "columns": ["N", "xi", "F(xi)", "E"],
        "results": [],
        "errors": [],
        "root": "0"
    }

    #configuración inicial
    datos = list()
    x = sp.Symbol('x')
    Fun = sympify(fx)
    DerF = sympify(df)


    xn = []
    derf = []
    xi = x0 # Punto de inicio
    f = Fun.evalf(subs={x: x0}) #función evaluada en x0
    derivada = DerF.evalf(subs={x: x0}) #función derivada evaluada en x0
    c = 0
    Error = 100
    xn.append(xi)

    try:
        datos.append([c, '{:^15.7f}'.format(x0), '{:^15.7f}'.format(f)])


        while Error > Tol and f != 0 and derivada != 0 and c < Niter:

            xi = xi-f/derivada
            derivada = DerF.evalf(subs={x: xi})
            f = Fun.evalf(subs={x: xi})
            xn.append(xi)
            c = c+1
            if tipo_error == "absoluto":
                Error = abs(xn[c]-xn[c-1])
            else:
                Error = abs(xn[c]-xn[c-1])/xn[c]
            derf.append(derivada)
            datos.append([c, '{:^15.7f}'.format(float(xi)), '{:^15.7E}'.format(
                float(f)), '{:^15.7E}'.format(float(Error))])

    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append(
                "Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))
        return output

    output["results"] = datos
    output["root"] = xi
    return output

def newtonView(request):
    context = {}
    datos = {}

    if request.method == 'POST':
        try:
            fx = request.POST["funcion"]
            derf = request.POST["derivada"]
            x0 = request.POST["punto_inicial"]
            X0 = float(x0)
            tol = request.POST["tolerancia"]
            Tol = float(tol)
            niter = request.POST["iteraciones"]
            Niter = int(niter)
            tipo_error = request.POST.get('tipo_error')

            datos = newton(X0, Tol, Niter, fx, derf, tipo_error)
        
            if "results" in datos:
                df = pd.DataFrame(datos["results"], columns=datos["columns"])
                
                x = sp.symbols('x')
                funcion_expr = sp.sympify(fx)

                xi_copy = X0
                xs_copy = X0  # Corrección: inicializamos xs_copy igual que xi_copy

                intervalo_x = np.arange(xi_copy - 10, xs_copy + 10, 0.1)  # Ajusta el intervalo si es necesario
                fx_func = sp.lambdify(x, funcion_expr, 'numpy')
                intervalo_y = fx_func(intervalo_x)
                intervalo_x_completo = np.arange(xi_copy - 10, xs_copy + 10, 0.1)
                intervalo_y_completo = fx_func(intervalo_x_completo)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=intervalo_x_completo, y=intervalo_y_completo, mode='lines', name='f(x)'))

                if datos.get("root") is not None:
                    fig.add_trace(go.Scatter(x=[float(datos["root"])], y=[0], mode='markers', name='Raíz hallada'))

                fig.update_layout(title=f'Función: {fx} en intervalo [{xi_copy - 10}, {xs_copy + 10}]',
                                xaxis_title='x',
                                yaxis_title='f(x)')

            plot_html = fig.to_html(full_html=False, default_height=500, default_width=700)

            dataframe_to_txt(df, f'Newton_txt {fx}')
            plot_to_png(fig,f'Newton {fx}')

            context = {'df': df.to_html(), 'plot_html': plot_html, 'mensaje': f'La solución es: {datos["root"]}'}

            return render(request, 'one_method.html', context)
        
        except Exception as e:
                context = {'error_message': f'Hubo un error en el metodo de Newton en: {str(e)}'}
                return render(request, 'error.html', context)

    return render(request, 'one_method.html')

def raicesMultiples(fx, x0, tol, niter, tipo_error):
    output = {
        "columns": ["iter", "xi", "f(xi)", "E"],
        "iterations": niter,
        "errors": [],
        "results": [],
        "root": "0"
    }
    
    # Configuraciones iniciales
    results = []
    x = sp.Symbol('x')
    ex = sympify(fx)
    
    d_ex = diff(ex, x)  # Primera derivada de fx
    d2_ex = diff(d_ex, x)  # Segunda derivada de fx

    ex_2 = ex.subs(x, x0).evalf()  # Funcion evaluada en x0
    d_ex2 = d_ex.subs(x, x0).evalf()  # Primera derivada evaluada en x0
    d2_ex2 = d2_ex.subs(x, x0).evalf()  # Segunda derivada evaluada en x0

    # Verificar si la primera o la segunda derivada son cero
    if d_ex2 == 0 or d2_ex2 == 0:
        output["errors"].append("Error: La primera o la segunda derivada evaluadas en x0 son cero.")
        print('entro')
        return output

    i = 0
    error = 1.0000
    results.append([i, '{:^15.7E}'.format(x0), '{:^15.7E}'.format(ex_2)])  # Datos con formato dado

    try:
        while (error > tol) and (i < niter):  # Se repite hasta que el intervalo sea lo pequeño que se desee
            if i == 0:
                ex_2 = ex.subs(x, x0).evalf()  # Funcion evaluada en valor inicial
            else:
                d_ex2 = d_ex.subs(x, x0).evalf()  # Primera derivada evaluada en x0
                d2_ex2 = d2_ex.subs(x, x0).evalf()  # Segunda derivada evaluada en x0

                # Verificar si la primera o la segunda derivada son cero en cada iteración
                if d_ex2 == 0 or d2_ex2 == 0:
                    output["errors"].append("Error: La primera o la segunda derivada evaluadas durante la iteración son cero.")
                    return output

                xA = x0 - (ex_2 * d_ex2) / ((d_ex2) ** 2 - ex_2 * d2_ex2)  # Método de Newton-Raphson modificado
                ex_A = ex.subs(x, xA).evalf()  # Funcion evaluada en xA

                if tipo_error == "absoluto":
                    error = Abs(xA - x0)
                else:
                    error = Abs(xA - x0) / Abs(xA)
                error = error.evalf()  # Se calcula el error

                ex_2 = ex_A  # Se establece la nueva aproximación
                x0 = xA

                results.append([i, '{:^15.7E}'.format(float(xA)), '{:^15.7E}'.format(float(ex_2)), '{:^15.7E}'.format(float(error))])
            i += 1
    except Exception as e:
        output["errors"].append(f"Error en los datos: {str(e)}")
        return output

    output["results"] = results
    output["root"] = x0
    return output

def raicesMultiplesView(request):
    context = {}
    datos = {}

    if request.method == 'POST':
        try:
            Fx = request.POST["funcion"]

            X0 = request.POST["punto_inicial"]
            X0 = float(X0)

            niter = request.POST["iteraciones"]
            Niter = int(niter)

            Tol = request.POST["tolerancia"]
            Tol = float(Tol)

            tipo_error = request.POST.get('tipo_error')
            datos = raicesMultiples(Fx,X0,Tol,Niter,tipo_error)


            if "results" in datos and datos["results"]:
                df = pd.DataFrame(datos["results"], columns=datos["columns"])
                
                x = sp.symbols('x')
                funcion_expr = sp.sympify(Fx)

                xi_copy = X0
                xs_copy = X0  # Corrección: inicializamos xs_copy igual que xi_copy

                intervalo_x = np.arange(xi_copy - 10, xs_copy + 10, 0.1)  # Ajusta el intervalo si es necesario
                fx_func = sp.lambdify(x, funcion_expr, 'numpy')
                intervalo_y = fx_func(intervalo_x)
                intervalo_x_completo = np.arange(xi_copy - 10, xs_copy + 10, 0.1)
                intervalo_y_completo = fx_func(intervalo_x_completo)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=intervalo_x_completo, y=intervalo_y_completo, mode='lines', name='f(x)'))

                if datos.get("root") is not None:
                    fig.add_trace(go.Scatter(x=[float(datos["root"])], y=[0], mode='markers', name='Raíz hallada'))

                fig.update_layout(title=f'Función: {Fx} en intervalo [{xi_copy - 10}, {xs_copy + 10}]',
                                xaxis_title='x',
                                yaxis_title='f(x)')

            if "errors" in datos and datos["errors"]:
                context = {'error_message': f'Hubo un error en el metodo de Newton en: {str(datos["errors"])}'}
                return render(request, 'error.html', context)
            
            plot_html = fig.to_html(full_html=False, default_height=500, default_width=700)

            dataframe_to_txt(df, f'RaicesMultiples_txt {Fx}')
            plot_to_png(fig,f'RaicesMultiples {Fx}')

            context = {'df': df.to_html(), 'plot_html': plot_html, 'mensaje': f'La solución es: {datos["root"]}'}

            return render(request, 'one_method.html', context)

        except Exception as e:
                context = {'error_message': f'Hubo un error en el metodo de Newton en: {str(e)}'}
                return render(request, 'error.html', context)
        
    return render(request, 'one_method.html')
    

def iterativos(request):
    if request.method == 'POST':
        tamaño=request.POST['numero']
        metodo=request.POST['metodo_iterativo']
        tol=request.POST['tol']
        niter=request.POST['niter']
        if metodo=="sor":
            w=request.POST['w']
        try:
            matriz=[]
            tamaño=int(tamaño)
            tol=float(tol)
            niter=int(niter)
            if metodo=="sor":
                w=float(w)
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
            if metodo=="jacobi":
                context=metodo_jacobi(nmatriz,nvectorx,nvectorb,tol,niter)
            elif metodo=="gauss_seidel":
                context=metodo_gauss_seidel(nmatriz,nvectorx,nvectorb,tol,niter)
            elif metodo=="sor":
                context=metodo_sor(nmatriz,nvectorx,nvectorb,tol,niter,w)

            dataframe_to_txt(pd.DataFrame(nmatriz),metodo+"_matrizA")
            dataframe_to_txt(pd.DataFrame(nvectorb),metodo+"_vectorb")
            dataframe_to_txt(context['tabla'],metodo+"_tabla")
            return render(request,'resultado_iterativo.html',context)

        except:
            context={'mensaje':'No se pudo realizar la operación'}
            return render(request,'resultado_iterativo.html',context)


def interpolacion(request):
    if request.method == 'POST':
        try:
            metodo_interpolacion = request.POST.get('metodo_interpolacion')
            x_values = request.POST.getlist('x[]')
            y_values = request.POST.getlist('y[]')

            # Convertir los valores de x_values y y_values a floats
            x_floats = [float(x) for x in x_values]
            y_floats = [float(y) for y in y_values]

            xi = x_floats
            fi = y_floats

            n = len(xi)
            if n<=1:
                context = {'error_message': f'Se deben ingresar 2 puntos o más'}
                return render(request, 'error.html', context)
            for i in range(n-1):
                if xi[i] >= xi[i+1]:
                    context = {'error_message': f'Los valores de x deben estar ordenados de menor a mayor'}
                    return render(request, 'error.html', context)

            if metodo_interpolacion == 'lagrange':
                try:
                    xi = x_floats
                    fi = y_floats

                    # PROCEDIMIENTO
                    # Polinomio de Lagrange
                    
                    x = sp.Symbol('x')
                    polinomio = 0
                    divisorL = np.zeros(n, dtype=float)
                    iteraciones = []

                    for i in range(n):
                        numerador = 1
                        denominador = 1
                        numerador_str = ""
                        denominador_str = ""

                        for j in range(n):
                            if j != i:
                                numerador *= (x - xi[j])
                                denominador *= (xi[i] - xi[j])
                                numerador_str += f"(x - {xi[j]})"
                                if denominador_str:
                                    denominador_str += "*"
                                denominador_str += f"({xi[i]} - {xi[j]})"

                        terminoLi = numerador / denominador
                        polinomio += terminoLi * fi[i]
                        divisorL[i] = denominador

                        Li_str = f"{numerador_str} / {denominador_str}"
                        
                        # Guardar la iteración actual
                        iteraciones.append({
                            'i': i,
                            'L_i': Li_str
                        })

                    # Convertir iteraciones a DataFrame
                    df_iteraciones = pd.DataFrame(iteraciones)

                    polisimple = polinomio.expand()

                    # Puntos para la gráfica
                    a = np.min(xi)
                    b = np.max(xi)

                    # Crear la función lambda para el polinomio simplificado
                    px = sp.lambdify(x, polisimple, 'numpy')

                    # Intervalo de valores para la gráfica
                    intervalo_x_completo = np.arange(a, b, 0.1)
                    intervalo_y_completo = px(intervalo_x_completo)

                    # Crear la gráfica
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=intervalo_x_completo, y=intervalo_y_completo, mode='lines', name='f(x)'))

                    # Configurar el diseño de la gráfica
                    fig.update_layout(title=f'Función: {polisimple}',
                                        xaxis_title='x',
                                        yaxis_title='f(x)')

                    # Convertir la gráfica a HTML
                    plot_html = fig.to_html(full_html=False, default_height=500, default_width=700)

                    # Guardar la gráfica como imagen
                    plot_to_png(fig, f'Lagrange_img{polisimple}')

                    # Convertir DataFrame a HTML
                    df_html = df_iteraciones.to_html(classes='table table-striped', index=False)
                    polisimple_latex = sp.latex(polisimple)

                    mensaje = 'Polinomio de Lagrange:'

                    context = {
                        'polinomio': polisimple_latex,
                        'df': df_html,
                        'mensaje': mensaje,
                        'plot_html': plot_html,
                    }

                    dataframe_to_txt(df_iteraciones, "lagrange_iteraciones")
                    text_to_txt(str(polisimple),"lagrange_funcion")
                    return render(request, 'one_method.html', context)
                except Exception as e:
                    context = {'error_message': f'Hubo un error con lagrange en: {str(e)}'}
                    return render(request, 'error.html', context)

            elif metodo_interpolacion == 'newton':
                try:
                    xi = x_floats
                    fi = y_floats

                    # Convertir X y Y a arreglos numpy
                    X = np.array(x_floats)
                    Y = np.array(y_floats)
                    n = len(X)

                    # Calcular la tabla de diferencias divididas
                    D = np.zeros((n, n))
                    D[:, 0] = Y.T
                    for i in range(1, n):
                        for j in range(n - i):
                            D[j, i] = (D[j + 1, i - 1] - D[j, i - 1]) / (X[j + i] - X[j])
                    
                    Coef = D[0, :]  # Coeficientes del polinomio de Newton

                    # Crear DataFrame para la tabla de diferencias divididas
                    diff_table_df = pd.DataFrame(D)
                    diff_table_df_to_html = diff_table_df.to_html(classes='table table-striped', index=False)

                    # Construir el polinomio de Newton
                    x = sp.Symbol('x')
                    newton_poly = Coef[0]
                    product_term = 1
                    for i in range(1, n):
                        product_term *= (x - X[i-1])
                        newton_poly += Coef[i] * product_term

                    # Crear DataFrame para el polinomio de Newton
                    newton_poly_df = pd.DataFrame({'Polinomio': [str(newton_poly)]})
                    newton_poly_df_to_html = newton_poly_df.to_html(classes='table table-striped', index=False)

                    # Evaluar el polinomio en un rango de x
                    x_vals = np.linspace(min(X), max(X), 100)
                    y_vals = [float(newton_poly.evalf(subs={x: val})) for val in x_vals]

                    fig = go.Figure()

                    # Agregar puntos originales
                    fig.add_trace(go.Scatter(x=X, y=Y, mode='markers', name='Puntos originales'))

                    # Agregar el polinomio de Newton
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Polinomio de Newton'))

                    fig.update_layout(title='Interpolación de Newton con Diferencias Divididas',
                                    xaxis_title='X',
                                    yaxis_title='Y')

                    # Graficar el polinomio de Newton
                    plot_html = fig.to_html(full_html=False, default_height=500, default_width=700)

                    context = {
                        'df': diff_table_df_to_html,
                        'df2': newton_poly_df_to_html,
                        'nombre_metodo': 'Newton Diferencias Divididas', 
                        'plot_html': plot_html
                    }

                    dataframe_to_txt(diff_table_df, 'Newton_diff_table')
                    dataframe_to_txt(newton_poly_df, 'Newton_poly')
                    plot_to_png(fig, 'Newton_DiffDiv_graph')
                    return render(request, 'one_method.html', context)
                except Exception as e:
                    context = {'error_message': f'Hubo un error con Newton Diferencias Divididas en: {str(e)}'}
                    return render(request, 'error.html', context)

            elif metodo_interpolacion == 'vandermonde':
                try:
                    x = sp.Symbol('x')
                    matriz_A = []
                    for i in range(n):
                        grado = n - 1
                        fila = []
                        for j in range(grado, -1, -1):
                            fila.append(xi[i] ** grado)
                            grado -= 1
                        matriz_A.append(fila)
                    vector_b = fi
                    nmatriz_A = np.array(matriz_A)
                    nvectorb = np.array(vector_b).reshape(-1, 1)
                    nvectora = np.linalg.inv(nmatriz_A) @ nvectorb
                    pol = ""
                    grado = n - 1
                    for i in range(n):
                        pol +=str(nvectora[i][0])
                        if i < n - 1:
                            if grado > 0:
                                pol += "*x**" + str(grado)
                            if (nvectora[i + 1][0] >= 0):
                                pol += "+"
                        grado -= 1


                    mA_html = pd.DataFrame(nmatriz_A).to_html()
                    vb_html = pd.DataFrame(nvectorb).to_html()
                    va_html = pd.DataFrame(nvectora).to_html()

                    fig=plot_fx_puntos(pol,xi[0],xi[-1],xi,fi,'punto dado')
                    plot_html = fig.to_html(full_html=False, default_height=500, default_width=700)
                    plot_to_png(fig,"vandermonde")
                    text_to_txt(pol,"vandermonde_pol")
                    
                    mensaje = "Se logro dar con una solucion"
                    context = {
                        'vandermonde': True,
                        'ma': mA_html,
                        'vb': vb_html,
                        'va': va_html,
                        'polinomio': pol,
                        'nombre_metodo': "Vandermonde",
                        'mensaje': mensaje,
                        'plot_html': plot_html
                    }
                    return render(request, 'one_method.html', context)
                except Exception as e:
                    context = {'error_message': f'Hubo un error con vandermonde: {str(e)}'}
                    return render(request, 'error.html', context)
                

            elif metodo_interpolacion == 'splinel':

                xi = x_floats
                fi = y_floats    

                # Convertir X y Y a arreglos numpy
                X = np.array(x_floats)
                Y = np.array(y_floats)
                n = len(X)
                
                # Variables para construir las funciones
                x = sp.Symbol('x')
                px_tabla = []
                coef_list = []

                # Calcular los coeficientes y las funciones lineales por tramos
                for i in range(1, n):
                    # Calcula la pendiente
                    numerador = Y[i] - Y[i-1]
                    denominador = X[i] - X[i-1]
                    m = numerador / denominador
                    
                    # Guarda los coeficientes (m, b) como float
                    coef_list.append([float(m), float(Y[i-1] - m * X[i-1])])
                    
                    # Construye la función lineal para el tramo
                    pxtramo = Y[i-1] + m * (x - X[i-1])
                    px_tabla.append(pxtramo)
                
                # Crear DataFrame para los coeficientes
                coef_df = pd.DataFrame(coef_list, columns=['Pendiente (m)', 'Intersección (b)'])
                coef_df_to_html = coef_df.to_html(classes='table table-striped', index=False)

                # Crear DataFrame para las funciones
                func_df = pd.DataFrame({'Función': [str(func) for func in px_tabla]})
                func_df_to_html = func_df.to_html(classes='table table-striped', index=False)

                # Evaluar las funciones en un rango de x
                funciones_evaluadas = []
                for i, func in enumerate(px_tabla):
                    x_vals = np.linspace(X[i], X[i+1], 100)
                    y_vals = [float(func.evalf(subs={x: val})) for val in x_vals]
                    funciones_evaluadas.append((x_vals, y_vals))
                
                fig = go.Figure()

                # Agregar puntos originales
                fig.add_trace(go.Scatter(x=X, y=Y, mode='markers', name='Puntos originales'))
                
                # Agregar las funciones por tramos
                for i, (x_vals, y_vals) in enumerate(funciones_evaluadas):
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f'Tramo {i+1}'))
                
                fig.update_layout(title='Spline Cubico por Tramos',
                                xaxis_title='X',
                                yaxis_title='Y')
                
                # Graficar las funciones por tramos
                plot_html = fig.to_html(full_html=False, default_height=500, default_width=700)

                context = {
                    'df': coef_df_to_html,
                    'df2': func_df_to_html,
                    'nombre_metodo': 'Spline Lineal', 
                    'plot_html': plot_html
                }

                dataframe_to_txt(coef_df, 'Spline_coef')
                dataframe_to_txt(func_df, 'Spline_func')
                plot_to_png(fig, f'Spline_Lineal_graph')
            elif metodo_interpolacion=="splinecu":
                x=xi
                y=fi
                A = np.zeros((4*(n-1), 4*(n-1)))
                b = np.zeros(4*(n-1))
                cua = np.square(x)
                cub = np.power(x, 3)
                c = 0
                h = 0
                d=3
                
                for i in range(n - 1):
                    A[i, c] = cub[i]
                    A[i, c + 1] = cua[i]
                    A[i, c + 2] = x[i]
                    A[i, c + 3] = 1
                    b[i] = y[i]
                    c += 4
                    h += 1
                
                c = 0
                for i in range(1, n):
                    A[h, c] = cub[i]
                    A[h, c + 1] = cua[i]
                    A[h, c + 2] = x[i]
                    A[h, c + 3] = 1
                    b[h] = y[i]
                    c += 4
                    h += 1
                
                c = 0
                for i in range(1, n - 1):
                    A[h, c] = 3 * cua[i]
                    A[h, c + 1] = 2 * x[i]
                    A[h, c + 2] = 1
                    A[h, c + 4] = -3 * cua[i]
                    A[h, c + 5] = -2 * x[i]
                    A[h, c + 6] = -1
                    b[h] = 0
                    c += 4
                    h += 1
                
                c = 0
                for i in range(1, n - 1):
                    A[h, c] = 6 * x[i]
                    A[h, c + 1] = 2
                    A[h, c + 4] = -6 * x[i]
                    A[h, c + 5] = -2
                    b[h] = 0
                    c += 4
                    h += 1
                
                A[h, 0] = 6 * x[0]
                A[h, 1] = 2
                b[h] = 0
                h += 1
                A[h, c] = 6 * x[-1]
                A[h, c + 1] = 2
                b[h] = 0
                val = np.linalg.inv(A).dot(b)
                Tabla = val.reshape((n - 1, d + 1))
                fxs=[]
                for i in range(n-1):
                    fx=""
                    for j in range(4):
                        fx+=str(Tabla[i,j])
                        if j<3:
                            fx+="*x"
                            if j<2:
                                fx+="**"+str(3-j)
                            if(Tabla[i,j+1]>=0):
                                fx+="+"
                    fxs.append(fx)
                tabla_df=pd.DataFrame(fxs)
                tabla_html=tabla_df.to_html(classes='table table-striped', index=False)
                fig=spline_plot(fxs,xi,fi)
                plot_html=fig.to_html(full_html=False, default_height=500, default_width=1200)
                coeficientes=pd.DataFrame(A)
                context={
                    'df':tabla_html,
                    'coef':coeficientes.to_html(classes='table table-striped', index=False),
                    'spline':True,
                    'plot_html':plot_html
                }
                dataframe_to_txt(coeficientes,"spline_cubico_coef")
                dataframe_to_txt(tabla_df,"spline_cubico_funciones")
                plot_to_png(fig,"spline_cubico_grafica")


        
        except Exception as e:
            context = {'error_message': f'Ocurrió un error en la obtención de los datos, llene todos los campos bien, numeros enteros o decimales'}
            return render(request, 'error.html', context)

    return render(request, 'one_method.html', context)


