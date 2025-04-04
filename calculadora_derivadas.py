import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

# Configuración de la página
st.set_page_config(page_title="Calculadora de derivadas", layout="wide")

# Estilo CSS personalizado
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1400px; /* Increased max width for better layout */
        margin: 0 auto;
    }
    h1 {
        color: white !important;
        background-color: #d90429 !important;
        padding: 1rem !important;
        border-radius: 5px !important;
        margin: -2rem -4rem 2rem -4rem !important;
        text-align: center !important;
        width: calc(100% + 8rem) !important;
    }
    h3 {
        color: #1E63B4;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .result-area {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .latex-output {
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    .button-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr); /* Adjusted to 5 columns for better spacing */
        gap: 20px; /* Increased gap for better spacing */
        margin-bottom: 20px; /* Added more margin */
    }
    .button-row {
        display: flex;
        flex-wrap: wrap;
        gap: 20px; /* Increased gap for better spacing */
        margin-bottom: 20px; /* Added more margin */
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.title("Calculadora avanzada de derivadas")

# Inicializar el estado de la sesión si no existe
if 'last_expression' not in st.session_state:
    st.session_state.last_expression = ""
if 'current_input' not in st.session_state:
    st.session_state.current_input = ""

def update_input(value):
    """Actualiza el input sin causar reruns innecesarios"""
    st.session_state.current_input = value

def add_to_input(symbol):
    """Función simplificada para añadir símbolos"""
    current = st.session_state.current_input
    st.session_state.current_input = current + symbol

def clear_input():
    """Función simplificada para limpiar el input"""
    st.session_state.current_input = ""

# Función para traducir expresiones matemáticas
def translate_math_expression(expr_str):
    # Reemplazar "sen" por "sin"
    expr_str = expr_str.replace("sen", "sin")
    # Reemplazar "ln" por "log"
    expr_str = expr_str.replace("ln(", "log(")
    # Reemplazar "e^" por "exp" y añadir el paréntesis de cierre
    expr_str = expr_str.replace("e^x", "exp(x)")
    if "e^(" in expr_str:
        expr_str = expr_str.replace("e^(", "exp(")
    elif "e^" in expr_str:
        # Para casos como e^2x o e^3x
        parts = expr_str.split("e^")
        new_parts = []
        for i, part in enumerate(parts):
            if i == 0:
                new_parts.append(part)
            else:
                # Encontrar el final de la expresión
                j = 0
                while j < len(part) and (part[j].isalnum() or part[j] in "+-*/^()"):
                    j += 1
                new_parts.append(f"exp({part[:j]}){part[j:]}")
        expr_str = "".join(new_parts)
    return expr_str

# Función para traducir de nuevo a notación amigable
def translate_back(expr_str):
    # Reemplazar "sin" por "sen" en la salida
    expr_str = expr_str.replace("sin", "sen")
    # Reemplazar "log" por "ln" en la salida
    expr_str = expr_str.replace("log", "ln")
    return expr_str

# Función para resolver la derivada
def solve_derivative(expr_str, var_str='x', order=1, with_respect_to_y=False, implicit=False):
    if not expr_str:
        return None, None, None, None, None, None
    
    try:
        # Configurar transformaciones personalizadas
        transformations = (standard_transformations + 
                         (implicit_multiplication_application,) + 
                         (convert_xor,))
        
        translated_expr = translate_math_expression(expr_str)
        x, y = sp.symbols('x y')
        
        # Verificar si es una constante numérica o pi
        try:
            # Primero intentamos evaluar como una expresión simbólica
            if translated_expr.lower() == 'pi':
                expr = sp.pi
            elif translated_expr == 'e':
                expr = sp.E
            else:
                # Intenta convertir a número
                const_val = float(translated_expr)
                expr = sp.Number(const_val)
            
            # La derivada de una constante siempre es 0
            derivative = sp.Number(0)
            
            # Convertir a LaTeX
            original_latex = sp.latex(expr)
            derivative_latex = sp.latex(derivative)
            
            # Traducir a notación amigable
            original_human = str(expr)
            derivative_human = str(derivative)
            
            return expr, derivative, original_latex, derivative_latex, original_human, derivative_human
            
        except ValueError:
            # No es una constante numérica, continuar con el proceso normal
            pass
        
        # Manejo especial para ecuaciones implícitas
        if implicit:
            # Separar la ecuación en lados izquierdo y derecho
            if '=' in translated_expr:
                left, right = translated_expr.split('=')
                # Mover todo a un lado de la ecuación
                translated_expr = f"({left})-({right})"
            
            # Parsear la expresión con ambas variables
            expr = parse_expr(translated_expr, transformations=transformations, 
                            local_dict={'x': x, 'y': y})
            
            if with_respect_to_y:
                # Derivada implícita respecto a y
                derivative = -sp.diff(expr, x) / sp.diff(expr, y)
            else:
                # Derivada implícita respecto a x
                derivative = -sp.diff(expr, y) / sp.diff(expr, x)
        else:
            # Para derivadas normales
            var = sp.symbols(var_str)
            expr = parse_expr(translated_expr, transformations=transformations, 
                            local_dict={'x': x})
            derivative = sp.diff(expr, var, order)
        
        # Convertir a LaTeX
        original_latex = sp.latex(expr)
        derivative_latex = sp.latex(derivative)
        
        # Traducir a notación amigable
        original_human = translate_back(str(expr))
        derivative_human = translate_back(str(derivative))
        
        return expr, derivative, original_latex, derivative_latex, original_human, derivative_human
    except Exception as e:
        st.error(f"Error al calcular la derivada: {str(e)}")
        return None, None, None, None, None, None

def is_constant(expr):
    """Verifica si la expresión es una constante"""
    try:
        return not expr.free_symbols
    except:
        return False

# Función para graficar la función y su derivada
def plot_function_and_derivative(expr, derivative):
    try:
        # Verificar si es una constante
        if is_constant(expr):
            st.info("La función es una constante. No se genera gráfica para constantes.")
            return None

        # Verificar si es una función compleja (composición de funciones)
        expr_str = str(expr)
        is_complex = ('log' in expr_str or 'ln' in expr_str) and ('sin' in expr_str or 'cos' in expr_str or 'tan' in expr_str)
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor('#000000')
        fig.patch.set_facecolor('#000000')
        
        x = sp.symbols('x')
        modules = ['numpy', {
            'log': lambda x: np.log(abs(x)) if abs(x) > 1e-10 else np.nan,
            'exp': np.exp,
            'E': np.e,
            'Abs': np.abs,
            'sign': np.sign,
            'sin': lambda x: np.sin(float(x)),
            'cos': lambda x: np.cos(float(x)),
            'tan': lambda x: np.tan(float(x)) if abs(np.cos(float(x))) > 1e-10 else np.nan
        }]
        
        f = sp.lambdify(x, expr, modules=modules)
        df = sp.lambdify(x, derivative, modules=modules)
        
        # Ajustar rango y número de puntos según complejidad
        if is_complex:
            # Reducir rango y puntos para funciones complejas
            x_vals = np.linspace(0.1, np.pi, 500)
            max_points = 100  # Limitar puntos válidos para funciones complejas
        elif 'tan' in expr_str:
            x = np.linspace(-2*np.pi, 2*np.pi, 1000)
            x_vals = x[abs(np.cos(x)) > 0.1]
        elif 'log' in expr_str or 'ln' in expr_str:
            x_vals = np.linspace(0.1, 5, 1000)  # Reducir rango para logaritmos
        elif 'exp' in expr_str or 'e^' in expr_str:
            x_vals = np.linspace(-3, 3, 1000)
        else:
            x_vals = np.linspace(-10, 10, 1000)

        y_vals = []
        dy_vals = []
        valid_x = []
        points_added = 0
        
        for x_val in x_vals:
            try:
                if is_complex and points_added >= max_points:
                    break
                
                y_val = float(f(x_val))
                dy_val = float(df(x_val))
                
                if (np.isfinite(y_val) and np.isfinite(dy_val) and 
                    abs(y_val) < 1e4 and abs(dy_val) < 1e4):
                    valid_x.append(x_val)
                    y_vals.append(y_val)
                    dy_vals.append(dy_val)
                    points_added += 1
                    
            except (ValueError, TypeError, ZeroDivisionError, RuntimeWarning):
                continue

        if valid_x:
            # Colores más visibles sobre fondo negro
            ax.plot(valid_x, y_vals, '#ff0066', label='Función f(x)', linewidth=2)
            ax.plot(valid_x, dy_vals, '#00ff66', label='Derivada f\'(x)', linewidth=2)
            
            # Solo calcular intersecciones para funciones simples
            if not is_complex:
                intersections = []
                last_y = None
                last_dy = None
                last_x = None
                
                for x_val, y_val, dy_val in zip(valid_x, y_vals, dy_vals):
                    if last_y is not None and last_dy is not None:
                        if (y_val - dy_val) * (last_y - last_dy) <= 0:
                            t = (last_y - last_dy) / ((last_y - last_dy) - (y_val - dy_val))
                            x_int = last_x + t * (x_val - last_x)
                            y_int = last_y + t * (y_val - last_y)
                            
                            if np.isfinite(x_int) and np.isfinite(y_int):
                                intersections.append((x_int, y_int))
                    
                    last_y = y_val
                    last_dy = dy_val
                    last_x = x_val
                
                if intersections:
                    intersect_x, intersect_y = zip(*intersections)
                    ax.plot(intersect_x, intersect_y, 'o', color='white', 
                           markerfacecolor='white', markeredgecolor='black',
                           markersize=8, label='Puntos de intersección')
                    
                    for i, (x_int, y_int) in enumerate(intersections):
                        ax.annotate(f'({x_int:.2f}, {y_int:.2f})',
                                   (x_int, y_int),
                                   textcoords="offset points",
                                   xytext=(0,10),
                                   ha='center',
                                   color='white',
                                   bbox=dict(facecolor='#000000', 
                                           edgecolor='white',
                                           alpha=0.7))
            
            # Configuración de la gráfica
            ax.grid(True, color='#333333', alpha=0.3)
            ax.axhline(y=0, color='#444444', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='#444444', linestyle='-', alpha=0.3)
            
            ax.set_xlabel('x', color='white', fontsize=12)
            ax.set_ylabel('y', color='white', fontsize=12)
            ax.set_title('Función y su derivada', color='white', fontsize=14)
            
            ax.tick_params(colors='white')
            
            ax.legend(facecolor='black', edgecolor='#666666')
            
            margin = (max(y_vals) - min(y_vals)) * 0.1
            ax.set_ylim(min(y_vals) - margin, max(y_vals) + margin)
            
            return fig
        else:
            st.warning("No se pueden graficar estas funciones en el rango especificado.")
            return None
    except Exception as e:
        st.error(f"Error al crear la gráfica: {str(e)}")
        return None

def get_derivative_rule(expr):
    """Retorna la regla de derivación en LaTeX con explicación más detallada"""
    try:
        expr_str = str(expr)
        rules = []

        # Reglas básicas
        if 'x' in expr_str:
            rules.append(r"$\text{Regla básica: } \frac{d}{dx}(x) = 1$")
        if any(c.isdigit() for c in expr_str):
            rules.append(r"$\text{Regla de la constante: } \frac{d}{dx}(c) = 0$")

        # Reglas de operaciones
        if '+' in expr_str or '-' in expr_str:
            rules.append(r"$\text{Regla de la suma/resta: } \frac{d}{dx}[f(x) \pm g(x)] = \frac{d}{dx}f(x) \pm \frac{d}{dx}g(x)$")
        if '*' in expr_str:
            rules.append(r"$\text{Regla del producto: } \frac{d}{dx}[u \cdot v] = u\frac{dv}{dx} + v\frac{du}{dx}$")
        if '/' in expr_str:
            rules.append(r"$\text{Regla del cociente: } \frac{d}{dx}\left(\frac{u}{v}\right) = \frac{v\frac{du}{dx} - u\frac{dv}{dx}}{v^2}$")

        # Reglas de potencias y exponenciales
        if '^' in expr_str or 'pow' in expr_str:
            rules.append(r"$\text{Regla de la potencia: } \frac{d}{dx}(x^n) = nx^{n-1}$")
        if 'exp' in expr_str or 'e^' in expr_str:
            rules.append(r"$\text{Regla exponencial: } \frac{d}{dx}(e^x) = e^x$")

        # Reglas trigonométricas
        if 'sin' in expr_str:
            rules.append(r"$\text{Regla para seno: } \frac{d}{dx}[\sen(x)] = \cos(x)$")
        if 'cos' in expr_str:
            rules.append(r"$\text{Regla para coseno: } \frac{d}{dx}[\cos(x)] = -\sen(x)$")
        if 'tan' in expr_str:
            rules.append(r"$\text{Regla para tangente: } \frac{d}{dx}[\tan(x)] = \sec^2(x)$")
        if 'csc' in expr_str:
            rules.append(r"$\text{Regla para cosecante: } \frac{d}{dx}[\csc(x)] = -\csc(x)\cot(x)$")
        if 'sec' in expr_str:
            rules.append(r"$\text{Regla para secante: } \frac{d}{dx}[\sec(x)] = \sec(x)\tan(x)$")
        if 'cot' in expr_str:
            rules.append(r"$\text{Regla para cotangente: } \frac{d}{dx}[\cot(x)] = -\csc^2(x)$")

        # Reglas para funciones hiperbólicas
        if 'sinh' in expr_str:
            rules.append(r"$\text{Regla para seno hiperbólico: } \frac{d}{dx}[\sinh(x)] = \cosh(x)$")
        if 'cosh' in expr_str:
            rules.append(r"$\text{Regla para coseno hiperbólico: } \frac{d}{dx}[\cosh(x)] = \sinh(x)$")
        if 'tanh' in expr_str:
            rules.append(r"$\text{Regla para tangente hiperbólica: } \frac{d}{dx}[\tanh(x)] = \sech^2(x)$")

        # Reglas para logaritmos
        if 'log' in expr_str:
            rules.append(r"$\text{Regla para logaritmo natural: } \frac{d}{dx}[\ln(x)] = \frac{1}{x}$")
        if 'log10' in expr_str:
            rules.append(r"$\text{Regla para logaritmo base 10: } \frac{d}{dx}[\log_{10}(x)] = \frac{1}{x\ln(10)}$")

        # Reglas para funciones compuestas
        if '(' in expr_str and ')' in expr_str:
            rules.append(r"$\text{Regla de la cadena: } \frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$")

        if not rules:
            rules.append(r"$\text{Regla básica de derivación}$")

        return "\n\n".join(rules)
    except:
        return r"$\text{No se pudo identificar la regla}$"

def get_step_by_step_derivation(expr, derivative):
    """Genera explicación paso a paso de la derivación"""
    try:
        steps = []
        expr_str = str(expr)
        
        # Identificar las partes de la expresión
        terms = expr.as_ordered_terms()
        
        for i, term in enumerate(terms, 1):
            step = {
                "term": sp.latex(term),
                "rule": get_derivative_rule(term),
                "derivative": sp.latex(sp.diff(term)),
                "explanation": explain_derivative_step(term)
            }
            steps.append(step)
        
        return steps
    except:
        return []

def explain_derivative_step(term):
    """Explica el proceso de derivación de un término"""
    try:
        term_str = str(term)
        if '^' in term_str:
            base, exp = term_str.split('^')
            return f"Aplicamos la regla de la potencia: multiplicamos por el exponente {exp} y reducimos la potencia en 1"
        elif 'sin' in term_str:
            return "La derivada del seno es coseno"
        elif 'cos' in term_str:
            return "La derivada del coseno es menos seno"
        # ... más explicaciones para otros casos
        return "Aplicamos la regla correspondiente"
    except:
        return "Proceso de derivación"

def get_derivative_steps(expr, derivative):
    """Genera un análisis detallado del proceso de derivación"""
    try:
        steps = []
        
        # Verificar si es una constante
        if is_constant(expr):
            step_info = {
                "original": sp.latex(expr),
                "parts": [],
                "rules": [{"name": "Regla de la constante", 
                          "latex": r"\frac{d}{dx}(c) = 0"}],
                "steps": [{"explanation": "Aplicamos la regla de la constante",
                          "general_rule": r"\frac{d}{dx}(c) = 0",
                          "application": "0"}],
                "result": "0"
            }
            steps.append(step_info)
            return steps

        # Descomponer la expresión en sus términos
        terms = expr.as_ordered_terms()
        
        for term in terms:
            step_info = {
                "original": sp.latex(term),
                "parts": [],
                "rules": [],
                "steps": [],
                "result": sp.latex(sp.diff(term))
            }
            
            # Analizar la estructura del término
            if term.is_Mul:  # Producto de factores
                factors = term.as_ordered_factors()
                step_info["parts"] = [sp.latex(f) for f in factors]
                for f in factors:
                    if f.has(sp.Symbol('x')):
                        rule = identify_rule(f)
                        step_info["rules"].append(rule)
                        step_info["steps"].append(explain_step(f, rule))
            else:
                rule = identify_rule(term)
                step_info["rules"].append(rule)
                step_info["steps"].append(explain_step(term, rule))
            
            steps.append(step_info)
        
        return steps
    except:
        return []

def identify_rule(expr):
    """Identifica la regla específica para un término"""
    expr_str = str(expr)
    if 'exp' in expr_str:
        return {"name": "Regla exponencial",
                "latex": r"\frac{d}{dx}e^x = e^x"}
    elif 'log' in expr_str:
        return {"name": "Regla logarítmica",
                "latex": r"\frac{d}{dx}\ln(x) = \frac{1}{x}"}
    if 'sin' in expr_str:
        return {"name": "Regla del seno", 
                "latex": r"\frac{d}{dx}\sen(x) = \cos(x)"}
    elif 'cos' in expr_str:
        return {"name": "Regla del coseno",
                "latex": r"\frac{d}{dx}\cos(x) = -\sen(x)"}
    elif 'tan' in expr_str:
        return {"name": "Regla de la tangente",
                "latex": r"\frac{d}{dx}\tan(x) = \sec^2(x)"}
    elif 'csc' in expr_str:
        return {"name": "Regla de la cosecante",
                "latex": r"\frac{d}{dx}\csc(x) = -\csc(x)\cot(x)"}
    elif 'sec' in expr_str:
        return {"name": "Regla de la secante",
                "latex": r"\frac{d}{dx}\sec(x) = \sec(x)\tan(x)"}
    elif 'cot' in expr_str:
        return {"name": "Regla de la cotangente",
                "latex": r"\frac{d}{dx}\cot(x) = -\csc^2(x)"}
    elif '^' in expr_str:
        n = expr.as_base_exp()[1]
        return {"name": "Regla de la potencia",
                "latex": r"\frac{d}{dx}x^{{{sp.latex(n)}}} = {sp.latex(n)}x^{{{sp.latex(n-1)}}}"}
    if expr.is_Number:
        return {"name": "Regla de la constante", "latex": r"\frac{d}{dx}(c) = 0"}
    if expr.is_Symbol:
        return {"name": "Regla de la variable", "latex": r"\frac{d}{dx}(x) = 1"}
    if expr.is_Add:
        return {"name": "Regla de la suma/resta", "latex": r"\frac{d}{dx}[f(x) \pm g(x)] = \frac{d}{dx}f(x) \pm \frac{d}{dx}g(x)"}
    if expr.is_Mul:
        return {"name": "Regla del producto", "latex": r"\frac{d}{dx}[u \cdot v] = u\frac{dv}{dx} + v\frac{du}{dx}"}
    if expr.is_Pow:
        base, exp = expr.as_base_exp()
        if base.is_Symbol and exp.is_Number:
            return {"name": "Regla de la potencia", "latex": rf"\frac{{d}}{{dx}}(x^{{{sp.latex(exp)}}}) = {sp.latex(exp)}x^{{{sp.latex(exp - 1)}}}"}
        return {"name": "Regla general de la potencia", "latex": r"\frac{d}{dx}(u^v) = v \cdot u^{v-1} \cdot \frac{du}{dx} + u^v \cdot \ln(u) \cdot \frac{dv}{dx}"}
    if 'sinh' in expr_str:
        return {"name": "Regla del seno hiperbólico",
                "latex": r"\frac{d}{dx}\sinh(x) = \cosh(x)"}
    elif 'cosh' in expr_str:
        return {"name": "Regla del coseno hiperbólico",
                "latex": r"\frac{d}{dx}\cosh(x) = \sinh(x)"}
    elif 'tanh' in expr_str:
        return {"name": "Regla de la tangente hiperbólica",
                "latex": r"\frac{d}{dx}\tanh(x) = \sech^2(x)"}
    elif 'asinh' in expr_str:
        return {"name": "Regla del arco seno hiperbólico",
                "latex": r"\frac{d}{dx}\sinh^{-1}(x) = \frac{1}{\sqrt{x^2 + 1}}"}
    elif 'acosh' in expr_str:
        return {"name": "Regla del arco coseno hiperbólico",
                "latex": r"\frac{d}{dx}\cosh^{-1}(x) = \frac{1}{\sqrt{x^2 - 1}}"}
    elif 'atanh' in expr_str:
        return {"name": "Regla del arco tangente hiperbólico",
                "latex": r"\frac{d}{dx}\tanh^{-1}(x) = \frac{1}{1 - x^2}"}
    return {"name": "Regla básica", "latex": r"\frac{d}{dx}x = 1"}

def explain_step(term, rule):
    """Explica el proceso de derivación paso a paso"""
    try:
        step = {
            "explanation": f"Aplicamos {rule['name']}",
            "general_rule": rule["latex"],
            "application": sp.latex(sp.diff(term))
        }
        if term.is_Mul:
            step["explanation"] += " y la regla del producto"
        return step
    except:
        return {"explanation": "Aplicamos la regla correspondiente", 
                "general_rule": "", 
                "application": ""}

# Modificar el manejo del input
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0

def reset_input():
    st.session_state.current_input = ""
    st.session_state.input_key += 1

# Crear columnas para el diseño
col1, col2 = st.columns([1.5, 1])  # Increased space for the input section

# Panel de opciones en la primera columna
with col2:
    st.subheader("Opciones de derivación")
    
    # Tipo de derivación
    derivative_type = st.radio(
        "Tipo de derivación:",
        ["Estándar", "Implícita"]
    )
    
    # Opciones adicionales
    if derivative_type == "Estándar":
        var_name = st.text_input("Variable de derivación:", "x")
        order = st.number_input("Orden de la derivada:", min_value=1, max_value=5, value=1)
        with_respect_to_y = False
        implicit = False
    else:  # Implícita
        with_respect_to_y = st.checkbox("Derivar respecto a y", False)
        order = st.number_input("Orden de la derivada:", min_value=1, max_value=3, value=1)
        var_name = "x" if not with_respect_to_y else "y"
        implicit = True

# Segunda columna para la entrada y salida
with col1:  # Adjusted to use the larger column
    st.subheader("Entrada de función")
    
    # Sección de botones para operadores - MEJORADO
    with st.container():
        st.markdown("<div class='button-section'>", unsafe_allow_html=True)
        
        # Fila 1: Operadores básicos
        cols1 = st.columns(5)  # Adjusted to 5 columns for better spacing
        
        # Definir los botones para la fila 1
        buttons_row1 = [
            {"label": "CLR", "value": "", "action": clear_input},
            {"label": "&#43;", "value": "+", "action": lambda: add_to_input("+")},
            {"label": "−", "value": "-", "action": lambda: add_to_input("-")},
            {"label": "×", "value": "*", "action": lambda: add_to_input("*")},
            {"label": "÷", "value": "/", "action": lambda: add_to_input("/")}
        ]
        
        # Renderizar botones de la fila 1
        for i, btn in enumerate(buttons_row1):
            with cols1[i]:
                btn_key = f"btn_{btn['label']}_{i}"
                if st.button(btn["label"], key=btn_key, use_container_width=True):
                    btn["action"]()
        
        # Fila 2: Potencias y funciones
        cols2 = st.columns(5)  # Adjusted to 5 columns for better spacing
        
        buttons_row2 = [
            {"label": "^", "value": "^", "action": lambda: add_to_input("^")},
            {"label": "√", "value": "sqrt(", "action": lambda: add_to_input("sqrt(")},
            {"label": "(", "value": "(", "action": lambda: add_to_input("(")},
            {"label": ")", "value": ")", "action": lambda: add_to_input(")")},
            {"label": "e", "value": "e", "action": lambda: add_to_input("e")}
        ]
        
        # Renderizar botones de la fila 2
        for i, btn in enumerate(buttons_row2):
            with cols2[i]:
                btn_key = f"btn_{btn['label']}_{i}"
                if st.button(btn["label"], key=btn_key, use_container_width=True):
                    btn["action"]()

        # Fila 3: Funciones trigonométricas
        cols3 = st.columns(5)  # Adjusted to 5 columns for better spacing
        
        trig_buttons = [
            {"label": "sen", "value": "sen(", "action": lambda: add_to_input("sen(")},
            {"label": "cos", "value": "cos(", "action": lambda: add_to_input("cos(")},
            {"label": "tan", "value": "tan(", "action": lambda: add_to_input("tan(")},
            {"label": "ln", "value": "ln(", "action": lambda: add_to_input("ln(")},
            {"label": "π", "value": "pi", "action": lambda: add_to_input("pi")}
        ]
        
        # Renderizar botones de la fila 3
        for i, btn in enumerate(trig_buttons):
            with cols3[i]:
                btn_key = f"btn_{btn['label']}_{i}"
                if st.button(btn["label"], key=btn_key, use_container_width=True):
                    btn["action"]()
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Campo de entrada de función simplificado
    placeholder_text = "Ejemplo: x^2 + sen(x) + ln(x)"
    expr_str = st.text_input("Ingresa la función a derivar:", 
                            value=st.session_state.current_input,
                            placeholder=placeholder_text,
                            key="function_input",
                            on_change=lambda: update_input(st.session_state.function_input))
    
    # Actualizar el estado de la sesión con el valor actual del campo de texto
    st.session_state.current_input = expr_str
    
    # Botones de acción
    col_calc, col_clear = st.columns([3, 1])
    with col_calc:
        calculate_btn = st.button("Calcular derivada", key="calculate_button")
    with col_clear:
        if st.button("Limpiar", key="clear_button"):
            clear_input()

# Área de resultados
st.subheader("Resultados")
result_container = st.container()

def simplify_function(expr):
    """Simplifica la función ingresada para evitar confusiones"""
    try:
        simplified_expr = sp.simplify(expr)
        return simplified_expr
    except:
        return expr

# Calcular la derivada cuando cambia la entrada o se presiona el botón
should_calculate = expr_str != st.session_state.last_expression or calculate_btn
if should_calculate:
    st.session_state.last_expression = expr_str

with result_container:
    if expr_str and should_calculate:
        # Calcular derivada
        expr, derivative, original_latex, derivative_latex, original_human, derivative_human = solve_derivative(
            expr_str, var_name, order, with_respect_to_y, implicit
        )
        
        if expr and derivative:
            # Mostrar la función original
            st.markdown("### Función Original")
            st.write("Esta es la función tal como fue ingresada:")
            st.latex(original_latex)
            
            # Mostrar la función simplificada
            simplified_expr = simplify_function(expr)
            if simplified_expr != expr:
                st.markdown("### Función Simplificada")
                st.write("Esta es la versión simplificada de la función:")
                st.latex(sp.latex(simplified_expr))
            
            # Mostrar la derivada
            st.markdown(f"### Derivada {'respecto a ' + var_name + ' ' if derivative_type == 'Implícita' else ''}de orden {order}")
            st.latex(derivative_latex)
            
            # Mostrar el procedimiento detallado
            st.markdown("### Procedimiento de Derivación")
            steps = get_derivative_steps(expr, derivative)
            
            with st.expander("Ver procedimiento detallado", expanded=True):
                st.write("**Función a derivar:**")
                st.latex(original_latex)
                
                if len(steps) > 1:
                    st.write("Aplicaremos la regla de la suma/resta: derivamos cada término por separado")
                
                for i, step in enumerate(steps, 1):
                    st.markdown(f"**Término {i}:**")
                    st.latex(step["original"])
                    
                    for j, (rule, explanation) in enumerate(zip(step["rules"], step["steps"]), 1):
                        st.write(f"Paso {j}: {explanation['explanation']}")
                        st.latex(explanation["general_rule"])
                        st.write("Aplicación:")
                        st.latex(explanation["application"])
                    
                    st.write("Resultado para este término:")
                    st.latex(step["result"])
                    st.markdown("---")
                
                st.write("**Resultado final:**")
                st.latex(derivative_latex)
            
            # Graficar
            if derivative_type == "Estándar":
                st.markdown("### Gráfica")
                if not is_constant(expr):
                    fig = plot_function_and_derivative(expr, derivative)
                    if fig:
                        st.pyplot(fig)
                else:
                    st.info(f"La función f(x) = {str(expr)} es una constante. Su derivada es 0 y no requiere gráfica.")

# Agregar instrucciones y ejemplos
st.markdown("---")
st.subheader("Instrucciones y ejemplos")

with st.expander("Ver instrucciones"):
    st.markdown("""
    ### Cómo usar la calculadora:
    
    1. **Ingresa tu función** en el campo de texto utilizando la notación matemática estándar o usa los botones para agregar símbolos.
    2. **Selecciona el tipo de derivación** (estándar o implícita).
    3. **Ajusta las opciones** según tus necesidades.
    4. **Haz clic en "Calcular derivada"** para ver los resultados.
    
    ### Notación soportada:
    
    - **Multiplicación implícita:** Puedes escribir `2x` en lugar de `2*x`
    - **Potencias:** Utiliza `^` para elevar, por ejemplo, `x^2`
    - **Funciones trigonométricas:** Puedes usar `sen(x)` en lugar de `sin(x)`
    - **Logaritmo natural:** Utiliza `ln(x)` en lugar de `log(x)`
    - **Raíz cuadrada:** Utiliza `sqrt(x)`
    
    ### Ejemplos de funciones:
    
    - `x^2 + 3x + 2`
    - `sen(x) + cos(x)`
    - `ln(x) + x^3`
    - `sqrt(x) * x^2`
    - `e^x + x^2`
    - `x^3 * sen(x)`
    
    ### Para derivadas implícitas:
    
    Ingresa una ecuación que contenga tanto `x` como `y`, por ejemplo:
    - `x^2 + y^2 = 25`
    - `sen(x*y) + x = y`
    - `x*y + y^2 = 4`
    """)

# Footer
st.markdown("---")
st.caption("Calculadora de Derivadas | Desarrollada con Streamlit, SymPy y Matplotlib")