# app.py

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Superstore – Descuentos y Rentabilidad",
    page_icon= "🏬",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("superstore.csv", encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"])

    df["Year"]  = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

    df["Profit Margin"] = df["Profit"] / df["Sales"]

    bins   = [0, 0.01, 0.2, 0.4, 0.6, 1]
    labels = ["0%", "0–20%", "20–40%", "40–60%", "60–100%"]
    df["Discount Bucket"] = pd.cut(
        df["Discount"], bins=bins, labels=labels, include_lowest=True
    )
    return df

df = load_data()

# Paleta personalizada
palette = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#D9534F"]
paleta_mismo_colores= ["#ADADAD", "#4A8FB0", "#156082"]

# ---- SIDEBAR: FILTROS ----
st.sidebar.title("Filtros")

years  = st.sidebar.multiselect(
    "Año", options=sorted(df["Year"].unique()),
    default=sorted(df["Year"].unique())
)

regions = st.sidebar.multiselect(
    "Región", options=sorted(df["Region"].unique()),
    default=sorted(df["Region"].unique())
)

segments = st.sidebar.multiselect(
    "Segmento", options=sorted(df["Segment"].unique()),
    default=sorted(df["Segment"].unique())
)

categories = st.sidebar.multiselect(
    "Categoría", options=sorted(df["Category"].unique()),
    default=sorted(df["Category"].unique())
)

discount_buckets = st.sidebar.multiselect(
    "Rango de descuento",
    options=df["Discount Bucket"].dropna().unique(),
    default=df["Discount Bucket"].dropna().unique()
)

# Aplicar filtros
mask = (
    df["Year"].isin(years) &
    df["Region"].isin(regions) &
    df["Segment"].isin(segments) &
    df["Category"].isin(categories)
)

df["Date"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()

# Discount Bucket puede tener NaN
filtered = df[mask]
if len(discount_buckets) > 0:
    filtered = filtered[filtered["Discount Bucket"].isin(discount_buckets)]

# Titulo
st.title("Impacto de descuentos en la rentabilidad de Superstore")

# ---- PESTAÑAS ----
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 KPIs",
    "🏷️ Categorías",
    "🔎 Subcategorías",
    "💸 Descuentos",
    "🕒 Evolución",
    "🛍️ Productos más vendidos"
])

# Tab 1
with tab1:

    st.subheader("Efecto global de la estrategia de descuentos")
    # Separador visual
    st.markdown("---")

    # Cálculos KPI
    total_sales  = filtered["Sales"].sum()
    total_profit = filtered["Profit"].sum()
    margin       = (total_profit / total_sales) if total_sales != 0 else 0
    high_disc_pct = (filtered["Discount"] >= 0.3).mean() if len(filtered) > 0 else 0
    #        FILA 1 (KPIs)
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Ventas totales", f"${total_sales:,.0f}")
        st.markdown(
            """
            **Volumen económico generado.**  
            Valores altos → fuerte actividad comercial.  
            Valores bajos → menos movimiento.
            """
        )

    with col2:
        st.metric("Utilidad total", f"${total_profit:,.0f}")
        st.markdown(
            """
            **Mide cuánto gana realmente la empresa después de costos.**  
            Baja utilidad con altas ventas → no se está vendiendo con buen margen.
            """
        )
    # Separador visual
    st.markdown("---")
    #        FILA 2 (KPIs)
    col3, col4 = st.columns(2)

    with col3:
        st.metric("Margen promedio", f"{margin*100:,.1f}%")
        st.markdown(
            """
            **Refleja cuánta utilidad se obtiene por cada dólar vendido.**  
            Margen bajo → descuentos agresivos o productos poco rentables.
            """
        )

    with col4:
        st.metric("% líneas con descuento ≥ 30%", f"{high_disc_pct*100:,.1f}%")
        st.markdown(
            """
            **Indica qué tan agresiva es la estrategia de descuentos.**  
            Un valor alto aumenta ventas, pero puede reducir la rentabilidad.
            """
        )

    # Separador visual
    st.markdown("---")

# Tab 2
with tab2:
    # ---- GRÁFICO 1: Sales & Profit por Categoría (barras) ----
    st.subheader("El volumen de ventas no garantiza utilidad: Furniture vende, pero no rentabiliza")

    cat_group = (
        filtered
        .groupby("Category")[["Sales", "Profit"]]
        .sum()
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        
        cat_sorted = cat_group.sort_values("Sales", ascending=False).reset_index(drop=True)
        # paleta_mismo_colores va de claro a oscuro; invertir para que el primero (mayor) sea el más fuerte
        colores_ordenados = paleta_mismo_colores[::-1]
        color_map = {
            cat: colores_ordenados[i] if i < len(colores_ordenados) else paleta_mismo_colores[0]
            for i, cat in enumerate(cat_sorted["Category"])
        }
        
        fig_cat_sales = px.bar(
            cat_group,
            x="Category", y="Sales",
            title="Ventas de mayor a menor",
            color="Category",
            color_discrete_map=color_map,
            category_orders={"Category": cat_sorted["Category"].tolist()}
        )
        
        fig_cat_sales.update_xaxes(title=None)

        fig_cat_sales.update_xaxes(showgrid=False, zeroline=False)
        fig_cat_sales.update_yaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=None
        )
        # Eliminar ruido leyenda
        fig_cat_sales.update_layout(showlegend=False)
        fig_cat_sales.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

        fig_cat_sales.update_traces(
            text=cat_group["Sales"].apply(lambda x: f"${x:,.0f}"),
            textposition="outside"
        )
        
        st.plotly_chart(fig_cat_sales, use_container_width=True)

    with col2:
        cat_sorted = cat_group.sort_values("Sales", ascending=False).reset_index(drop=True)
        # paleta_mismo_colores va de claro a oscuro; invertir para que el primero (mayor) sea el más fuerte
        colores_ordenados = paleta_mismo_colores[::-1]
        color_map = {
            cat: colores_ordenados[i] if i < len(colores_ordenados) else paleta_mismo_colores[0]
            for i, cat in enumerate(cat_sorted["Category"])
        }
        
        fig_cat_profit = px.bar(
            cat_group,
            x="Category", y="Profit",
            title="Pero, ¿en cuáles realmente ganamos dinero?",
            color="Category",
            color_discrete_map=color_map,
            category_orders={"Category": cat_sorted["Category"].tolist()}
            
        )
        # Borrar eje x
        fig_cat_profit.update_xaxes(title=None)

        fig_cat_profit.update_xaxes(showgrid=False, zeroline=False)
        fig_cat_profit.update_yaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=None
        )
        fig_cat_profit.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        
        # Eliminar ruido leyenda
        fig_cat_profit.update_layout(showlegend=False)

        fig_cat_profit.update_traces(
            text=cat_group["Profit"].apply(lambda x: f"${x:,.0f}"),
            textposition="outside"
        )

        st.plotly_chart(fig_cat_profit, use_container_width=True)


    st.markdown("""
    El primer gráfico muestra qué categorías generan los mayores niveles de ventas. 
    Sin embargo, el segundo gráfico revela que **vender más no siempre significa ganar más**.
    Office Suplies y Furniture parecen tener ventas similares pero al evaluar la utilidad, Furniture es la menos rentable. 
    """)


    # Línea separadora
    st.markdown("---")

# Tab 3:
with tab3:

    # Agrupamos y ordenamos por utilidad
    sub_group = (
        filtered
        .groupby("Sub-Category")[["Sales", "Profit"]]
        .sum()
        .reset_index()
    )

    sub_group_sorted = sub_group.sort_values("Profit")

    # Detectar la subcategoría con mayor pérdida
    worst_sub = sub_group_sorted.iloc[0]["Sub-Category"]
    worst_profit = sub_group_sorted.iloc[0]["Profit"]

    # Subheader dinámico
    if worst_profit < 0:
        st.subheader(f"La mayor pérdida proviene de **{worst_sub}**")
    else:
        st.subheader("Ninguna subcategoría presenta pérdidas con estos filtros")

    # ---- GRÁFICO (igual que antes) ----
    sub_group_sorted["resalte"] = "Estable"
    sub_group_sorted.loc[sub_group_sorted["Profit"] < 0, "resalte"] = "En pérdida"

    fig_sub = px.bar(
        sub_group_sorted,
        x="Profit",
        y="Sub-Category",
        orientation="h",
        color="resalte",
        color_discrete_map={
            "Estable": paleta_mismo_colores[0],
            "En pérdida": "#1F77B4"
        }
    )

    fig_sub.update_layout(
        title=f"Distribución de utilidades por subcategoría",
        xaxis_title="Utilidad",
        yaxis_title="",
        legend_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    fig_sub.add_vline(x=0, line_width=1, line_dash="dash", line_color="#78909C")

    fig_sub.update_xaxes(showgrid=False, zeroline=False, tickformat="$,.0f")
    fig_sub.update_yaxes(showgrid=False, zeroline=False)

    st.plotly_chart(fig_sub, use_container_width=True)

    st.info("📘 Las subcategorías que aparecen bajo cero representan oportunidades clave para mejorar la rentabilidad.")

    st.markdown("---")

# Tab 4:
with tab4:
    # ---- GRÁFICO 3: Discount vs Profit (scatter) ----
    st.subheader("A mayor descuento, hay mayor probabilidad de pérdida por línea de pedido")
    fig_bubble = px.scatter(
        filtered,
        x="Discount",
        y="Profit",
        size="Sales",            # tamaño de la burbuja = monto vendido
        color="Category",        # puedes cambiar a "Sub-Category" si quieres más detalle
        hover_data=[
            "Order ID",
            "Product Name",
            "Category",
            "Sub-Category",
            "Sales",
            "Profit",
            "Discount"
        ],
        size_max=30,             # tamaño máximo de las burbujas (ajústalo si se ve muy grande/pequeño)
        title="Relación negativa entre descuento y utilidad",
        labels={
            "Discount": "Descuento",
            "Profit": "Utilidad (Profit)",
            "Sales": "Ventas (Sales)"
        }
    )

    # Añadir línea en 0
    fig_bubble.add_hline(
        y=0,
        line_dash="dash",
        line_color="#999999",
        line_width=1
    )

    # Etiquetas negrita
    fig_bubble.update_xaxes(
        showgrid=False,
        zeroline=False,
        title_font=dict(size=14, weight="bold")
    )
    fig_bubble.update_yaxes(
        showgrid=False,
        zeroline=False,
        title_font=dict(size=14, weight="bold")
    )

    st.plotly_chart(fig_bubble, use_container_width=True)

    st.info("Un mayor nivel de descuento no implica necesariamente una utilidad mayor. Las ventas grandes pueden generar pérdidas como se puede ver al aplicar diferentes filtros.")

    # Línea separadora
    st.markdown("---")
    # ---- GRÁFICO 4: Profit Margin por Discount Bucket (boxplot) ----
    st.subheader("A partir del 20–40% de descuento, la rentabilidad se deteriora de forma sistemática")
    # Orden de los buckets
    bucket_order = ["0%", "0–20%", "20–40%", "40–60%", "60–100%"]

    # Asegurar tipo texto para que Plotly no lo trate como número/código
    box_df = filtered.dropna(subset=["Discount Bucket"]).copy()
    box_df["Discount Bucket"] = box_df["Discount Bucket"].astype(str)
    box_df["Discount Bucket"] = box_df["Discount Bucket"].replace({"60–80%": "60–100%"})

    # Contar observaciones por bucket
    counts = (
        box_df.groupby("Discount Bucket")
        .size()
        .reindex(bucket_order)
        .dropna()
        .astype(int)
    )

    fig_box = px.violin(
        box_df,
        x="Discount Bucket",
        y="Profit Margin",
        category_orders={"Discount Bucket": bucket_order},
        color="Discount Bucket",
        color_discrete_sequence=palette,
        box=True,
        points="suspectedoutliers"
    )
    # Eje X categórico y con orden fijo
    fig_box.update_xaxes(type="category", categoryorder="array", categoryarray=bucket_order)

    fig_box.update_xaxes(range=[-0.4, len(bucket_order)-0.4])

    # Línea 0
    fig_box.add_hline(y=0, line_dash="dash", line_color="#78909C", line_width=1)

    # Mejorar legibilidad: recortar extremos
    # Ajusta si quieres ver todo:
    q1, q99 = box_df["Profit Margin"].quantile([0.01, 0.99])
    fig_box.update_yaxes(range=[q1, q99])


    # Títulos y estilo
    fig_box.update_xaxes(
        title="Rango de descuento",
        title_font=dict(size=16, family="Arial"),
        showgrid=False,
        zeroline=False
    )
    fig_box.update_yaxes(
        title="Margen de utilidad",
        title_font=dict(size=16, family="Arial"),
        showgrid=False,
        zeroline=False
    )

    fig_box.update_layout(
        title="A mayor descuento, mayor dispersión y más riesgo de margen negativo",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )
    fig_box.update_xaxes(type="category")
    fig_box.update_layout(
        height=550
    )
    fig_box.update_layout(margin=dict(t=140, b=60, l=90, r=60))

    # % de margen negativo por bucket
    neg_rate = (
        (box_df["Profit Margin"] < 0)
        .groupby(box_df["Discount Bucket"])
        .mean()
        .reindex(bucket_order)
    )

    # más espacio arriba para colocar textos fuera del área de trazado
    fig_box.update_layout(
        margin=dict(t=190, b=60, l=60, r=40)  # ↑ aumenta espacio superior
    )

    # anotaciones fuera del plot
    for b, n in counts.items():
        r = float(neg_rate.loc[b]) if b in neg_rate.index and pd.notna(neg_rate.loc[b]) else None

        if r is None:
            label = "Sin datos"
        elif r == 1:
            label = "Todas las ventas pierden dinero"
        elif r == 0:
            label = "0% ventas con pérdida"
        else:
            label = f"Ventas con pérdida: {r:.1%}"

        fig_box.add_annotation(
            x=b,
            y=1.20,              # posición fija arriba del gráfico
            xref="x",
            yref="paper",
            text=f"n={n}<br>{label}",
            showarrow=False,
            align="center",
            font=dict(size=10),
            yanchor="top"
        )


    st.plotly_chart(fig_box, use_container_width=True)

    # ---- STORYTELLING debajo del gráfico ----
    st.markdown("Los descuentos agresivos pueden elevar el volumen vendido, pero erosionan el margen.")
    st.info("""
    📦 Los rangos bajos (0%–20%) mantienen una rentabilidad más predecible y saludable con una dispersión reducida y un riesgo mínimo de generar pérdidas. Esto sugiere que las ventas en estos rangos se realizan bajo una estructura de precios saludable y controlada.  
    📦 A partir del rango 20–40%, la situación cambia de forma drástica. La distribución del margen se desplaza por debajo del punto de equilibrio
    En los rangos de descuento más agresivos (40–60% y 60–100%) la totalidad de las ventas se realiza con pérdidas, además de mostrar una mayor dispersión y profundidad negativa del margen.""")
    # Línea separadora
    st.markdown("---")

# Tab 5:
with tab5:
    ## ---- GRÁFICO 4: Evolución mensual (línea) ----
    st.subheader("Vender más a lo largo del tiempo no ha garantizado mayor rentabilidad")

    # Agrupar usando la nueva columna Date
    time_group = (
        filtered
        .groupby("Date")[["Sales", "Profit"]]
        .sum()
        .reset_index()
        .sort_values("Date")
    )
    # Encontrar el máximo de ventas
    max_sales_idx = time_group["Sales"].idxmax()
    max_sales_date = time_group.loc[max_sales_idx, "Date"]
    max_sales_value = time_group.loc[max_sales_idx, "Sales"]

    # --- Ventas mensuales ---
    fig_time_sales = px.line(
        time_group,
        x="Date", y="Sales",
        title="Evolución de ventas",
        markers=True,
    )

    fig_time_sales.add_annotation(
        x=max_sales_date,
        y=max_sales_value,
        text=f"Máx: ${max_sales_value:,.0f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#264653",
        ax=0,
        ay=-40,
        bgcolor="#E9C46A",
        bordercolor="#264653",
        borderwidth=1,
        font=dict(size=11, color="#264653", weight="bold")
    )

    fig_time_sales.update_xaxes(
        showgrid=False,
        zeroline=False,
        title="",
        title_font=dict(size=14, family="Arial", weight="bold"),
        tickformat="%b %Y"   # <-- Muestra "Jan 2015", "Feb 2016", etc.
    )

    fig_time_sales.update_yaxes(
        showgrid=False,
        zeroline=False,
        title="",
        title_font=dict(size=14, family="Arial", weight="bold")
    )

    fig_time_sales.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=18, family="Arial", weight="bold")
        
    )
    # --- Utilidad mensual ---
    fig_time_profit = px.line(
        time_group,
        x="Date", y="Profit",
        title="Evolución de utilidad",
        markers=True,
    )
    # Encontrar el máximo de utilidad
    max_profit_idx = time_group["Profit"].idxmax()
    max_profit_date = time_group.loc[max_profit_idx, "Date"]
    max_profit_value = time_group.loc[max_profit_idx, "Profit"]

    # Añadir anotación en el punto máximo
    fig_time_profit.add_annotation(
        x=max_profit_date,
        y=max_profit_value,
        text=f"Máx: ${max_profit_value:,.0f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#264653",
        ax=0,
        ay=-40,
        bgcolor="#E9C46A",
        bordercolor="#264653",
        borderwidth=1,
        font=dict(size=11, color="#264653", weight="bold")
    )

    fig_time_profit.update_xaxes(
        showgrid=False,
        zeroline=False,
        title="",
        title_font=dict(size=14, family="Arial", weight="bold"),
        tickformat="%b %Y"
    )

    fig_time_profit.update_yaxes(
        showgrid=False,
        zeroline=False,
        title="",
        title_font=dict(size=14, family="Arial", weight="bold")
    )

    fig_time_profit.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=18, family="Arial", weight="bold")
    )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_time_sales, use_container_width=True)
    with col4:
        st.plotly_chart(fig_time_profit, use_container_width=True)

    st.info("""Los márgenes son más estables en descuentos bajos, pero se vuelven volátiles y negativos aproximadamente a partir del 40%.
    Los descuentos agresivos aumentan ventas, pero sacrifican rentabilidad.""")

    # Línea separadora
    st.markdown("---")

# Tab 6 (treemap)
with tab6:
    # ---- GRÁFICO 5: Top 10 productos por venta (treemap) ----
    prod_group = (
        filtered
        .groupby(["Category", "Sub-Category", "Product Name"])[["Sales", "Profit"]]
        .sum()
        .reset_index()
        .sort_values("Sales", ascending=False)
        .head(50)
    )
    # --- Título dinámico del Treemap según los filtros ---
    # Calcular la utilidad total por categoría en el conjunto filtrado del treemap
    cat_profit = (
        prod_group.groupby("Category")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )
    # Si no hay datos, evitar error y usar un título neutral
    if len(cat_profit) == 0:
        titulo_treemap = "Rendimiento por producto"
    else:
        top_cat = cat_profit.index[0]
        titulo_treemap = f"La rentabilidad se concentra en {top_cat}"
    # Subtitulo
    st.subheader(titulo_treemap)

    fig_tree = px.treemap(
        prod_group,
        path=["Category", "Sub-Category", "Product Name"],
        values="Sales",
        color="Profit",
        color_continuous_scale="Blues",
        title="Productos que más venden y su utilidad"
    )
    st.plotly_chart(fig_tree, use_container_width=True)
    st.markdown("Los tonos azules oscuros indican la mayor utilidad de las subcategorias.")

    # Storytelling
    # Detectar categoría más rentable dentro del filtrado actual
    top_cat = (
        prod_group.groupby("Category")["Profit"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    # Detectar subcategoría más rentable
    top_sub = (
        prod_group.groupby("Sub-Category")["Profit"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    # Detectar producto con mayor utilidad
    top_product_row = prod_group.loc[prod_group["Profit"].idxmax()]
    top_product = top_product_row["Product Name"]
    top_product_profit = top_product_row["Profit"]

    # Storytelling dinámico
    st.info(
        f"Con los filtros aplicados:\n\n"
        f"La categoría con mayor utilidad es **{top_cat}**. "
        f"La subcategoría más rentable es **{top_sub}**.\n\n"

        f"💡 El producto con mejor desempeño es **{top_product}**, "
        f"con una utilidad aproximada de **${top_product_profit:,.0f}**, "
        f"superando ampliamente al resto del catálogo.\n\n"

        f" **Conclusión estratégica:** Incluso con los filtros aplicados, "
        f"queda claro que la utilidad proviene principalmente de unos pocos productos. "
        f"Esto sugiere que la empresa depende de un conjunto reducido de ítems fuertes, "
        f"mientras que la mayoría aporta muy poco en comparación."
    )
