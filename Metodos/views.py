from django.shortcuts import render
import numpy as np
import pandas as pd
import sympy as sp
from sympy import *
from django.utils.safestring import mark_safe
import plotly.graph_objects as go
from .metodos.iterativos import metodo_gauss_seidel,metodo_jacobi,metodo_sor
from .utiles.saver import dataframe_to_txt,plot_to_png

def home(request):
    return render(request, 'All_methods.html')


def reglaFalsaView(request):
    datos = ()
    if request.method == 'POST':
        fx = request.POST["funcion"]

        x0 = request.POST["a"]
        X0 = float(x0)

        xi = request.POST["b"]
        Xi = float(xi)

        tol = request.POST["tolerancia"]
        Tol = float(tol)

        niter = request.POST["iteraciones"]
        Niter = int(niter)

        datos = reglaFalsa(X0, Xi, Niter, Tol, fx)
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

        dataframe_to_txt(df, f'ReglaFalsa {fx}')
        plot_to_png(fig,f'ReglaFalsa {fx}')

    if datos:
        context = {'df': df.to_html(), 'plot_html': plot_html}
        return render(request, 'regla_falsa.html', context)

    return render(request, 'regla_falsa.html')

def reglaFalsa(a, b, Niter, Tol, fx):

    output = {
        "columns": ["iter", "a", "xm", "b", "f(xm)", "E"]
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

                error = Abs(xm-xm0)
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


def puntoFijo(X0, Tol, Niter, fx, gx):

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

            error = abs(xA - xP)  # Se calcula el error

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
        fx = request.POST["funcion-F"]
        gx = request.POST["funcion-G"]

        x0 = request.POST["vInicial"]
        X0 = float(x0)

        tol = request.POST["tolerancia"]
        Tol = float(tol)

        niter = request.POST["iteraciones"]
        Niter = int(niter)

        datos = puntoFijo(X0,Tol,Niter,fx,gx)
        df = pd.DataFrame(datos["results"], columns=datos["columns"])
        print(df)

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
        context = {'df': df.to_html(), 'plot_html': plot_html}
        return render(request, 'one_method.html', context)

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
            return render(request,'resultado_iterativo.html',context)

        except:
            context={'mensaje':'No se pudo realizar la operación'}
            return render(request,'resultado_iterativo.html',context)

def interpolacion(request):
    if request.method == 'POST':
        metodo_interpolacion = request.POST.get('metodo_interpolacion')
        x_values = request.POST.getlist('x[]')
        y_values = request.POST.getlist('y[]')

        # Convertir los valores de x_values y y_values a floats
        x_floats = [float(x) for x in x_values]
        y_floats = [float(y) for y in y_values]

        if metodo_interpolacion == 'lagrange':
            xi = x_floats
            fi = y_floats

            # PROCEDIMIENTO
            # Polinomio de Lagrange
            n = len(xi)
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
                    'Iteración (i)': i,
                    'L_i': Li_str
                })

            # Convertir iteraciones a DataFrame
            df_iteraciones = pd.DataFrame(iteraciones)

            # Simplificar el polinomio
            polisimple = polinomio.expand()

            # Para evaluación numérica
            px = sp.lambdify(x, polisimple)

            # Puntos para la gráfica
            muestras = 101
            a = np.min(xi)
            b = np.max(xi)
            pxi = np.linspace(a, b, muestras)
            pfi = px(pxi)

            # Convertir DataFrame a HTML
            df_html = df_iteraciones.to_html(classes='table table-striped', index=False)
            polisimple_latex = sp.latex(polisimple)

            
            mensaje = 'Polinomio de Lagrange:'

            context = {
                'polinomio': polisimple_latex,
                'df': df_html,
                'mensaje': mensaje
            }
            
            dataframe_to_txt(df_iteraciones, f'Lagrange')

            return render(request, 'one_method.html', context)



        elif metodo_interpolacion == 'newton':
            pass

        elif metodo_interpolacion == 'vandermonde':
            xi = x_floats
            fi = y_floats

            # PROCEDIMIENTO
            # Polinomio de Lagrange
            n = len(xi)
            x = sp.Symbol('x')
            matriz_A=[]
            for i in range(n):
                grado=n-1
                fila=[]
                for j in range(grado,-1,-1):
                    fila.append(xi[i]**grado)
                    grado-=1
                matriz_A.append(fila)
            vector_b=fi
            nmatriz_A=np.array(matriz_A)
            nvectorb = np.array(vector_b).reshape(-1, 1)
            nvectora =  np.linalg.inv(nmatriz_A) @ nvectorb
            pol="El polinomio resultante es: "
            grado=n-1
            for i in range(n):
                pol+=str(nvectora[i][0])
                if i<n-1:
                    if grado>0:
                        pol+="x**"+str(grado)
                    if(nvectora[i+1][0]>=0):
                        pol+="+"
                grado-=1
                
            
            mA_html = pd.DataFrame(nmatriz_A).to_html()
            vb_html = pd.DataFrame(nvectorb).to_html()
            va_html = pd.DataFrame(nvectora).to_html()

            mensaje="Se logro dar con una solucion"
            context = {
                'vandermonde':True,
                'ma':mA_html,
                'vb':vb_html,
                'va':va_html,
                'polinomio': pol,
                'nombre_metodo': "Vandermonde",
                'mensaje': mensaje
            }
            return render(request, 'one_method.html', context)

            



    return render(request, 'one_method.html')





        