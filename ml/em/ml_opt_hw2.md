{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": [
        "XqXA1Q3ybAGA",
        "NAMqkJQBiaTI",
        "Dw-8p1MRm3Uk",
        "rcLGeMU0vm36",
        "B1oWGkCU6Ed2"
      ]
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# Convexity"
      ],
      "metadata": {
        "id": "-pdbqdnAa9eT"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 1"
      ],
      "metadata": {
        "id": "XqXA1Q3ybAGA"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Show that this function is convex.: $$\n",
        "f(x, y, z) = z \\log \\left(e^{\\frac{x}{z}} + e^{\\frac{y}{z}}\\right) + (z - 2)^2 + e^{\\frac{1}{x + y}}\n",
        "$$ where the function $f : \\mathbb{R}^3 \\to \\mathbb{R}$ has its domain defined as: $$\n",
        "\\text{dom } f = \\{ (x, y, z) \\in \\mathbb{R}^3 : x + y > 0, \\, z > 0 \\}.\n",
        "$$"
      ],
      "metadata": {
        "id": "Pm_49X6_bGQB"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Сначала докажем, что область определения - выпуклое множество. $x+y > 0$ - это выпуклое  множество:\n",
        "$$\\forall \\theta \\in [0,1], \\forall (x_1, y_1), (x_2, y_2) \\in S: \\theta x_1 + \\theta y_1 + (1-\\theta) x_2 + (1-\\theta)y_2 > \\theta0 + (1-\\theta)0 = 0$$\n",
        "\n",
        "$z>0$ - выпуклое множество, аналогичным образом. Область определение - их пересечение, а пересечение выпуклых множеств - выпукло.\n",
        "\n",
        "Вспомним, что сумма выпуклых функций - выпуклая функция. Поэтому нам достаточно доказать, что каждое из слагаемых - выпуклое.\n",
        "\n",
        "1. $(z-2)^2$ - парабола, известная выпуклая функция, но для точности воспользуемся необходимым и достаточным условием выпуклости для дважды дифф. функций: $$\\frac{(z-2)^2}{\\delta z \\delta z} = 2 > 0$$\n",
        "Функция более того, сильно выпуклая.\n",
        "2. Так как $z>0$, то $g(x,y,z) = z\\log(e^{\\frac{x}{z}} + e^{\\frac{y}{z}})$ - является результатом операции, сохраняющей выпуклость, $f(x,y) = \\log(e^x + e^y)$,  а именно: $g(x,y,z) = zf(\\frac{x}{z}, \\frac{y}{z})$. Тогда достаточно доказать, что  $f(x,y) = \\log(e^x + e^y)$ - выпуклая. Это log-sum-exp известная выпуклая функция, но для точности, аналогично воспользуемся условием.\n",
        "$$\\nabla f = \\begin{pmatrix} \\frac{e^x}{e^x + e^y} \\\\ \\frac{e^y}{e^x + e^y} \\end{pmatrix}$$\n",
        "$$\\nabla^2 f = \\begin{pmatrix} \\frac{e^x e^y}{(e^x + e^y)^2} & \\frac{e^x e^y}{(e^x + e^y)^2} \\\\ \\frac{e^x e^y}{(e^x + e^y)^2} & \\frac{e^x e^y}{(e^x + e^y)^2} \\end{pmatrix}$$\n",
        "Критерий Сильвестра. Первый минор: $\\frac{e^x e^y}{(e^x + e^y)^2} > 0$. Второй минор: $|\\nabla^2 f| = 0$, т.к. равные столбцы. Все главные миноры неотрицательны - матрица положительно полуопределена. По условию для дважды дифференцируемых - это выпуклая матрица.\n",
        "3. $e^x$ - выпуклая, возрастающая функция. Тогда композиция:$\\exp(\\frac{1}{x+y})$ будет выпуклой, если $\\frac{1}{x+y}$ - выпукла и её область значений содержится в области определения экспоненты. Последний факт доказан по умолчанию, так как область определения экспоненты - множество вещественных чисел, а область значения гиперболы (при $x+y >0$)- множество вещественных чисел больше 0. Докажем, что функция выпукла через условие.\n",
        "$$\\nabla f = \\begin{pmatrix} -\\frac{1}{(x+y)^2} \\\\ -\\frac{1}{(x+y)^2}\\end{pmatrix}$$\n",
        "$$ \\nabla^2 f =  \\begin{pmatrix} \\frac{4x + 4y}{(x+y)^3} & \\frac{4x + 4y}{(x+y)^3} \\\\\\frac{4x + 4y}{(x+y)^3} & \\frac{4x + 4y}{(x+y)^3}\\end{pmatrix}$$\n",
        "Аналогично прошлому пункту $\\frac{4x + 4y}{(x+y)^3} > 0$ при ($x+y>0$), а $|\\nabla^2 f| = 0$, так как одинаковые столбцы. Гессиан положительно полуопределен, функция выпукла.\n",
        "\n",
        "Итого все слагаемые - выпуклы, функция - выпукла."
      ],
      "metadata": {
        "id": "bfiakqvGbG79"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 2"
      ],
      "metadata": {
        "id": "NAMqkJQBiaTI"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "The center of mass of a body is an important concept in physics (mechanics). For a system of material points with masses $m_i$ and coordinates $x_i$, the center of mass is given by: $$\n",
        "x_c = \\frac{\\sum_{i=1}^k m_i x_i}{\\sum_{i=1}^k m_i}\n",
        "$$ The center of mass of a body does not always lie inside the body. For example, the center of mass of a doughnut is located in its hole. Prove that the center of mass of a system of material points lies in the convex hull of the set of these points."
      ],
      "metadata": {
        "id": "A1QumDQPkh5d"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Как мы знаем из эмпирических исследований, тел с отрицательной массой не существует. Из этого делаем вывод $\\forall i: m_i > 0$. Выполним преобразования и тем самым покажем, что центр масс лежит в выпуклой оболочке точек: $conv(𝑆) = \\sum_{i=1}^k \\theta_i x_i | x_i \\in S, \\sum_{i=1}^k \\theta_i = 1, \\theta_i \\geq 0$\n",
        "$$x_c = \\frac{\\sum_{i=1}^k m_ix_i}{\\sum_{i=1}^k m_i} = \\sum_{i=1}^k \\frac{m_i}{\\sum_{i=1}^k m_i}x_i$$\n",
        "Заметим, что при положительных массах: $\\frac{m_i}{\\sum_{i=1}^k m_i} > 0$, и $\\sum_{j=1}^k\\frac{m_j}{\\sum_{i=1}^k m_i} = \\frac{\\sum_{j=1}^k m_j}{\\sum_{i=1}^k m_i} = 1$.\n",
        "Из этого видим: $\\frac{m_i}{\\sum_{i=1}^k m_i}$ принадлежит множеству возможных $\\theta$. Тем самым:\n",
        "$x_c \\in conv(S)$."
      ],
      "metadata": {
        "id": "DKNWNXDQki9I"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 3"
      ],
      "metadata": {
        "id": "Dw-8p1MRm3Uk"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Show, that $\\mathbf{conv}\\{xx^\\top: x \\in \\mathbb{R}^n, \\Vert x\\Vert  = 1\\} = \\{A \\in \\mathbb{S}^n_+: \\text{tr}(A) = 1\\}$."
      ],
      "metadata": {
        "id": "a2dFXjN8nAPl"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Покажем, что левое множество - подмножество правого и наоборот, тем самым они равны. Пусть $A \\in conv(xx^T : x \\in \\mathbb{R}^n, ||x|| = 1)$:\n",
        "$$\\theta_i \\geq 0, \\sum_{i=1}^k \\theta_i = 1:A = \\sum_{i=1}^k \\theta_i x_ix_i^T$$\n",
        "Докажем, что $A\\in \\mathbb{S}_+^n$:\n",
        "$$\\forall y \\in \\mathbb{R}^n / \\{0\\}^n : y^T A y = y^T \\left[ \\sum_{i=1}^k \\theta_i x_ix_i^T \\right] y = \\sum_{i=1}^k \\theta_i y^T x_ix_i^T y = \\sum_{i=1}^k \\underset{\\geq 0}{\\theta_i} \\underset{\\geq 0}{<x_i,y>^2} \\geq 0 \\to A\\in \\mathbb{S}_+^n$$\n",
        "Докажем, что $tr(A) = 1$:\n",
        "$$ tr(A) = tr(\\sum_{i=1}^k \\theta_i x_ix_i^T) = \\sum_{i=1}^k \\theta_i tr(x_ix_i^T) = \\sum_{i=1}^k \\theta_i <x_i, x_i> =\\sum_{i=1}^k \\theta_i ||x_i||^2 = \\sum_{i=1}^k \\theta_i = 1$$\n",
        "\n",
        "Тем самым: $A \\in conv(xx^T : x \\in \\mathbb{R}^n, ||x|| = 1) \\to A \\in \\{A \\in \\mathbb{S}^n_+: \\text{tr}(A) = 1\\}$. Предположим, что $A \\in \\{A \\in \\mathbb{S}^n_+: \\text{tr}(A) = 1\\}$:\n",
        "\n",
        "Воспользуемся спектральным разложением: $$∃ Q, Λ: A = Q Λ Q^T$$ Такие, что Q - ортономированная матрица собственных векторов A, $Λ$ - матрица собственных значений. Тогда:\n",
        "$$A = Q \\Lambda Q^T = \\sum_{i=1}^n \\sum^n_{j=1} \\begin{cases} q_iq_j^T\\lambda_i | i=j \\\\ 0\\end{cases} = \\sum_{i=1}^n \\lambda_i q_i q_i^T$$\n",
        "Причем, $||q_i|| = 1$, так как $Q$ - ортонормированная. И известно, что $$\\sum_{i=1}^n \\lambda_i = tr(A) =1$$ И наконец, так как матрица положительно полуопределенная, то $\\forall i : \\lambda_i \\geq 0$.\n",
        "Тем самым: $$A \\in conv(xx^T : x \\in \\mathbb{R}^n, ||x|| = 1)$$\n",
        "\n",
        "Показали то, что предполагали."
      ],
      "metadata": {
        "id": "PmmK29vZnCxA"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 4"
      ],
      "metadata": {
        "id": "rcLGeMU0vm36"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Prove that the set of $\\{x \\in \\mathbb{R}^2 \\mid e^{x_1}\\le x_2\\}$ is convex."
      ],
      "metadata": {
        "id": "ej2m9bPCvx4b"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Посмотрим на это с другой стороны, заданное множество является надграфиком $e^x$:\n",
        "$$ \\text{epi } e^x = \\{(x_1, x_2) \\in \\mathbb{R}^2 | e^{x_1} \\leq x_2\\}$$\n",
        "\n",
        "Как мы знаем необходимым и достаточным условием выпуклости функции является выпуклость надграфика, соответсвенно из выпуклости функции следует выпуклость надграфика. Экспонетна - выпуклая функция, следовательно заданное множество - выпуклое."
      ],
      "metadata": {
        "id": "OZjA_51PvyPv"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 5"
      ],
      "metadata": {
        "id": "B1oWGkCU6Ed2"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Consider the function $f(x) = x^d$, where $x \\in \\mathbb{R}_{+}$. Fill the following table with ✅ or ❎. Explain your answers (with proofs)."
      ],
      "metadata": {
        "id": "bQK3H0pVAIpy"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "| d | Convex | Concave | Strictly Convex | $\\mu$-strongly convex|\n",
        "| ------------------------- | --- | --- | --- | --- |\n",
        "|-2, x $\\in \\mathbb{R}_{++}$|$\\checkmark$    |×    |$\\checkmark$    |×    |\n",
        "|-1, x $\\in \\mathbb{R}_{++}$|$\\checkmark$    |×    |$\\checkmark$    |×    |\n",
        "|0                          |$\\checkmark$    |$\\checkmark$    |×    |×    |\n",
        "|0.5                        |×    |$\\checkmark$    |×    |×    |\n",
        "|1                          |$\\checkmark$    |$\\checkmark$    |×    |×    |\n",
        "|$\\in$ (1; 2)               |$\\checkmark$    |×    |$\\checkmark$    |×    |\n",
        "|2                          |$\\checkmark$    |×    |$\\checkmark$    |$\\checkmark$    |\n",
        "|$> 2$                      |$\\checkmark$    |×    |$\\checkmark$    |×    |\n"
      ],
      "metadata": {
        "id": "rt-v9_M5AMWe"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Я пользуюсь признаком строгой выпуклости через градиент. Так как один из семинаристов в чате говорил, что мы можем его использовать."
      ],
      "metadata": {
        "id": "z94-PGKKB6TS"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "1. $-2, x \\in \\mathbb{R}_{++}$:\n",
        "$$f''(x) = 6x^{-4} > 0$$\n",
        "По необходимому и достаточному условию (далее просто н.д.у.) это строго выпуклая функция, однако не $\\mu$ сильно, так как $\\forall \\mu ∃x_0 \\forall x > x_0: 6x^{-4} < \\mu$. В силу того, что $\\lim_{x\\to \\infty} 6x^{-4} = 0$.\n",
        "\n",
        "2. $-1, x \\in \\mathbb{R}_{++}$:\n",
        "$$f''(x) = 2x^{-3} > 0$$\n",
        "По н.д.у. строго выпуклая. Не $\\mu$ сильно по той же причине.$\\lim_{x\\to \\infty} 2x^{-3} = 0$\n",
        "\n",
        "3. $x^0$:\n",
        "$$f''(x) = 0$$\n",
        "$$(-f(x))'' = 0$$\n",
        "По н.д.у функция и её отрицание - выпуклы. Следовательно функция выпукла и вогнута. Остальные следуют из равенства второй производной нулю.\n",
        "\n",
        "4. $x^{0.5}$:\n",
        "$$f''(x) = -\\frac{1}{4}x^{-1.5} < 0 \\to (-f(x))'' > 0$$\n",
        "По н.д.у. отрицание функции - выпукло. Следовательно функция вогнута. Так как вторая производная самой функции строго меньше 0, то функция не выпукла. Важно отметить, что в 0 у функции нет производной. Докажем, что она вогнута на всём множестве $\\mathbb{R}_+$. Возьмем точку 0, и любую другую $x$ не равную ей:\n",
        "$$\\theta \\in [0,1]:f((1-\\theta) 0 + \\theta x) = \\theta^{0.5}x^{0.5} \\geq \\theta x^{0.5} = \\theta(f(x)) + (1-\\theta)f(0)$$\n",
        "Функция вогнута.\n",
        "\n",
        "5. $x^1$:\n",
        "$$f''(x) = 0$$\n",
        "$$(-f(x))'' = 0$$\n",
        "Аналогично позапрошлому пункту.\n",
        "\n",
        "6. $\\forall p \\in (1,2), x^{p}$:\n",
        "$$f''(x) = \\underset{>0}{p(p-1)}x^{\\underset{<0}{p-2}} > 0$$\n",
        "Как и в первых двух пунктах по н.д.у. строго выпукла на $\\mathbb{R}_{++}$. По той же причине не $\\mu$ сильно. Проверим строгую выпуклость на $\\mathbb{R}_+$ признаком первого порядка:\n",
        "$$\\forall x >0: f(x) > f(0) + f'(0)(x-0)$$\n",
        "$$x^p > 0 + 0x = 0$$\n",
        "\n",
        "7. $x^2$:\n",
        "$$f''(x) = 2$$\n",
        "\n",
        "По н.д.у. это 2-сильно выпуклая функция.\n",
        "\n",
        "8. $\\forall p > 2: x^p$\n",
        "$$f''(x) =\\underset{>0}{p(p-1)}x^{\\underset{>0}{p-2}}$$\n",
        "Для любого $x>0$ выполняется строгое неравенство. (да я только сейчас понял, что можно доказывать проще) Это выполняется на любом открытом множестве (внутренности) в $\\mathbb{R}_+$.  Следовательно это строго выпуклая функция. Не является сильно выпуклым, так как $\\forall \\mu ∃x_0 \\forall 0\\leq x < x_0 : p(p-1)x^{p-2} < \\mu$. Это верно в силу: $\\lim_{x \\to 0} x^{p-2} = 0$\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "eK_UqbpQU5pv"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 6"
      ],
      "metadata": {
        "id": "sJmLm-apclSS"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Prove that the entropy function, defined as $$\n",
        "f(x) = -\\sum_{i=1}^n x_i \\log(x_i),\n",
        "$$ with $\\text{dom}(f) = \\{x \\in \\mathbb{R}^n_{++} : \\sum_{i=1}^n x_i = 1\\}$, is strictly concave."
      ],
      "metadata": {
        "id": "xLz4lRmTct6D"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Для начала докажем, что если функция строго выпуклая на множестве, то она строго выпуклая на любом его выпуклом подмножестве. $s \\subseteq S$:\n",
        "$$\\forall x_1 \\neq x_2 \\in S, \\forall \\theta \\in (0, 1): f(\\theta x_1 +(1-\\theta)x_2) < \\theta f(x_1) + (1-\\theta)f(x_2) \\to \\\\ \\forall x_1 \\neq x_2 \\in s \\subseteq S, \\forall \\theta \\in (0, 1): f(\\theta x_1 +(1-\\theta)x_2) < \\theta f(x_1) + (1-\\theta)f(x_2)$$\n",
        "Так как:\n",
        "$$\\forall x_1, x_2 \\in s \\subseteq S , \\forall \\theta \\in [0, 1]: \\theta x_1 +(1-\\theta)x_2 \\in s ⊆ S$$\n",
        "\n",
        "Покажем, что область определения выпукла:\n",
        "$$\\forall x, y \\in \\text{dom} (f),  \\forall \\theta \\in [0, 1]: \\sum_{i=1}^n \\theta x_i +  \\sum_{i=1}^n (1-\\theta) y_i = \\theta \\sum_{i=1}^n  x_i +  (1-\\theta)\\sum_{i=1}^n  y_i = \\theta + (1-\\theta) = 1 \\\\ \\to \\theta x + (1-\\theta)y \\in \\text{dom} (f)$$\n",
        "Действительно выпукла, чтобы доказать, что $f$ строго вогнута, покажем, что $(-f)$ строго выпукла, и пользуясь доказанным утверждением, докажем это на $\\mathbb{R}_{++}$.\n",
        "\n",
        "Запишем гессиан функции:\n",
        "$$\\nabla^2 (-f) = \\left(\\begin{matrix}\n",
        "\\frac{1}{x_1} & \\dots & 0 \\\\\n",
        "\\vdots & \\ddots & \\vdots \\\\\n",
        "0 & \\dots & \\frac{1}{x_n}\n",
        "\\end{matrix}\\right)$$\n",
        "Легко видеть, что любой главный минор:\n",
        "$$M_k = ∏_{i=1}^k \\frac{1}{x_i} > 0$$\n",
        "По н.д.у. строго выпукла. Соответственно по ранее доказанному строго выпукла и на $\\text{dom} (f)$. Соотвественно f строго вогнута."
      ],
      "metadata": {
        "id": "VwUn7iohcujw"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 7"
      ],
      "metadata": {
        "id": "KKHXeIFpR-8P"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Show that the maximum of a convex function $f$ over the polyhedron $P = \\text{conv}\\{v_1, \\ldots, v_k\\}$ is achieved at one of its vertices, i.e., $$\n",
        "\\sup_{x \\in P} f(x) = \\max_{i=1, \\ldots, k} f(v_i).\n",
        "$$\n",
        "A stronger statement is: the maximum of a convex function over a closed bounded convex set is achieved at an extreme point, i.e., a point in the set that is not a convex combination of any other points in the set. (you do not have to prove it). Hint: Assume the statement is false, and use Jensen’s inequality."
      ],
      "metadata": {
        "id": "1nLVjYTYS5VW"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Докажем более строгое утверждение. Пусть действительно максимум не достигается на экстремальной точке, а на какой-то выпуклой комбинации. Возьмем базис из экстремальных точек (ни одна из них не предствима как комбинация остальных), так что: $$\\text{conv} \\{v_1, \\dots, v_n\\} = P$$\n",
        "Возьмем любую из точек максимума, по предположению:\n",
        "$$\\exists (\\theta_1, \\dots, \\theta_n):\\\\ \\theta_i \\geq 0, \\sum_{i=1}^n \\theta_i = 1 \\\\ \\exists i, j, i\\neq j : \\theta_i, \\theta_j > 0 \\\\ v_{max} = \\sum_{i=1}^n \\theta_i v_i$$\n",
        "Выпишем для него неравенство Йенсена и выполним преобразования:\n",
        "$$f(v_{max}) = f(\\sum_{i=1}^n \\theta_i v_i) \\leq \\sum_{i=1}^n \\theta_i f(v_i)$$\n",
        "Возьмем:\n",
        "$$\\bar v = \\underset{i\\in\\{1, \\dots, n\\}: \\theta_i > 0}{\\text{argmax}} f(v_i)$$\n",
        "Тогда:\n",
        "$$f(v_{max}) \\leq \\sum_{i=1}^n \\theta_i f(v_i) \\leq \\sum_{i=1}^n \\theta_i f(\\hat v) =  f(\\hat v)$$\n",
        "$f(\\hat v)$ - один из максимумов, это экстремальная точка по построению - противоречие."
      ],
      "metadata": {
        "id": "mtnVO9KVS5v8"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 8"
      ],
      "metadata": {
        "id": "s10oKzByXJA_"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Show, that the two definitions of $\\mu$-strongly convex functions are equivalent:\n",
        "\n",
        "1. $f(x)$ is $\\mu$-strongly convex $\\iff$ for any $x_1, x_2 \\in S$ and $0 \\le \\lambda \\le 1$ for some $\\mu > 0$: $$\n",
        "f(\\lambda x_1 + (1 - \\lambda)x_2) \\le \\lambda f(x_1) + (1 - \\lambda)f(x_2) - \\frac{\\mu}{2} \\lambda (1 - \\lambda)\\|x_1 - x_2\\|^2\n",
        "$$\n",
        "2. $f(x)$ is $\\mu$-strongly convex $\\iff$ if there exists $\\mu>0$ such that the function $f(x) - \\dfrac{\\mu}{2}\\Vert x\\Vert^2$ is convex."
      ],
      "metadata": {
        "id": "Qh6_DVjia_EL"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "$2 \\to 1$: $g(x) = f(x)- \\dfrac{\\mu}{2}\\Vert x\\Vert^2$ выпишем неравенство для функции из определения выпуклости, $\\forall x_1, x_2 \\in S$ и $0 \\le \\lambda \\le 1$:\n",
        "$$g(\\lambda x_1 + (1-\\lambda)x_2) \\leq \\lambda g(x_1) + (1-\\lambda)g(x_2) \\to \\\\\n",
        "f(\\lambda x_1 + (1-\\lambda)x_2) - \\frac{\\mu}{2} ||\\lambda x_1 + (1-\\lambda)x_2||^2 \\leq \\lambda f(x_1) + (1-\\lambda)f(x_2) - \\lambda  \\frac{\\mu}{2} ||x_1||^2 - (1- \\lambda) \\frac{\\mu}{2}  ||x_2||^2 \\to \\\\ f(\\lambda x_1 + (1-\\lambda)x_2) \\leq \\lambda f(x_1) + (1-\\lambda)f(x_2) + \\frac{\\mu}{2} ||\\lambda x_1 + (1-\\lambda)x_2||^2 - \\lambda  \\frac{\\mu}{2} ||x_1||^2 - (1- \\lambda) \\frac{\\mu}{2}  ||x_2||^2 $$\n",
        "Так как базовая часть неравенства нас не интересует более, рассмотрим конкретно:\n",
        "$$\\frac{\\mu}{2} ||\\lambda x_1 + (1-\\lambda)x_2||^2 - \\lambda  \\frac{\\mu}{2} ||x_1||^2 - (1- \\lambda) \\frac{\\mu}{2}  ||x_2||^2 \\underset{\\mu > 0}{\\leftrightarrow} \\\\  ||\\lambda x_1 + (1-\\lambda)x_2||^2 - \\lambda  ||x_1||^2 - (1- \\lambda) ||x_2||^2  = \\lambda^2 ||x_1||^2 + 2\\lambda (1-\\lambda)<x_1, x_2> + (1-\\lambda)^2 ||x_2||^2 - \\lambda  ||x_1||^2 - (1- \\lambda) ||x_2||^2 = -\\lambda(1-\\lambda)||x_1||^2 + 2\\lambda (1-\\lambda)<x_1, x_2> - \\lambda(1-\\lambda)||x_2||^2  = -\\lambda (1-\\lambda) (||x_1||^2 - 2<x_1, x_2> + ||x_2||^2) =  -\\lambda (1-\\lambda) || x_1 - x_2||^2$$\n",
        "Вернемся к неравенству:\n",
        "$$f(\\lambda x_1 + (1-\\lambda)x_2) \\leq \\lambda f(x_1) + (1-\\lambda)f(x_2) + \\frac{\\mu}{2} ||\\lambda x_1 + (1-\\lambda)x_2||^2 - \\lambda  \\frac{\\mu}{2} ||x_1||^2 - (1- \\lambda) \\frac{\\mu}{2}  ||x_2||^2  = \\lambda f(x_1) + (1-\\lambda)f(x_2) - \\frac{\\mu}{2}\\lambda (1-\\lambda) || x_1 - x_2||^2$$\n",
        "Что и требовалось.\n",
        "\n",
        "$1 \\to 2$:\n",
        "$$f(\\lambda x_1 + (1-\\lambda)x_2) \\leq \\lambda f(x_1) + (1-\\lambda)f(x_2) - \\frac{\\mu}{2}\\lambda (1-\\lambda) || x_1 - x_2||^2 = \\lambda f(x_1) + (1-\\lambda)f(x_2) + \\frac{\\mu}{2} ||\\lambda x_1 + (1-\\lambda)x_2||^2 - \\lambda  \\frac{\\mu}{2} ||x_1||^2 - (1- \\lambda) \\frac{\\mu}{2}  ||x_2||^2 \\to \\\\ f(\\lambda x_1 + (1-\\lambda)x_2) - \\frac{\\mu}{2} ||\\lambda x_1 + (1-\\lambda)x_2||^2 \\leq \\lambda f(x_1) + (1-\\lambda)f(x_2) - \\lambda  \\frac{\\mu}{2} ||x_1||^2 - (1- \\lambda) \\frac{\\mu}{2}  ||x_2||^2 \\to g(\\lambda x_1 + (1-\\lambda)x_2) \\leq \\lambda g(x_1) + (1-\\lambda)g(x_2)$$\n",
        "Что и требовалось."
      ],
      "metadata": {
        "id": "TZZ6khHla_cC"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Optimality conditions. KKT. Duality"
      ],
      "metadata": {
        "id": "992ARZGjh3hU"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "In this section, you can consider either the arbitrary norm or the Euclidian norm if nothing else is specified."
      ],
      "metadata": {
        "id": "Ip_Xewe2h8R-"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 1"
      ],
      "metadata": {
        "id": "xErulvVUF1AH"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "$$\n",
        "\\begin{split}\n",
        "& x^2 + 1 \\to \\min\\limits_{x \\in \\mathbb{R} }\\\\\n",
        "\\text{s.t. } & (x-2)(x-4) \\leq 0\n",
        "\\end{split}\n",
        "$$\n",
        "\n",
        "**1.** Give the feasible set, the optimal value, and the optimal solution."
      ],
      "metadata": {
        "id": "Si4E0kJdFF3u"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Найдем множество допустимых значений:\n",
        "$$(x-2)(x-4) \\leq 0 \\to x \\in [2,4]$$\n",
        "\n",
        "Далее найдем оптимальное значение и решение. Я мог бы выписать лагранжиан, но это будет излишне. Проще понять, что $x^2 + 1$ - возрастающая функция на области допустим значений, следовательно решением будет левая граница множества. Тем более, что в прошлом задании мы показали, что оптимум выпуклой функции на выпуклой оболочке - одна из краевых точек.\n",
        "\n",
        "$$x_{min} = 2 \\to y_{min} = 4 + 1 = 5$$"
      ],
      "metadata": {
        "id": "wbsUELJcFekv"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**2.** Plot the objective $x^2 +1$ versus $x$. On the same plot, show the feasible set, optimal point, and value, and plot the Lagrangian $L(x,\\mu)$ versus $x$ for a few positive values of $\\mu$. Verify the lower bound property ($p^* \\geq \\inf_x L(x, \\mu)$for $\\mu \\geq 0$). Derive and sketch the Lagrange dual function $g$.\n",
        "\n"
      ],
      "metadata": {
        "id": "C6t5dqcPFGdk"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Заранее запишем лагранжиан и проверим свойство нижней границы:\n",
        "\n",
        "$$\\mu \\geq 0:L(x, \\mu) = x^2 + 1 + \\mu(x-2)(x-4) = (1+ \\mu)x^2 - 6\\mu x + 1 + 8\\mu$$\n",
        "\n",
        "$$\\inf_xL(x,\\mu):\\\\\n",
        "\\begin{cases} \\frac{dL}{dx} = 2(1 + \\mu) x - 6\\mu = 0 \\\\ \\frac{dL}{dxdx} = 2(1 + \\mu) > 0 \\end{cases}\\to x_{inf} = \\frac{3\\mu}{1+\\mu} \\to \\\\ \\inf_xL(x,\\mu) = \\frac{9\\mu^2}{1+\\mu} - \\frac{18\\mu^2}{1+\\mu} + 1 + 8\\mu = - \\frac{9\\mu^2}{1+\\mu} + 1 + 8\\mu = - \\mu + 10 - \\frac{9}{1 + \\mu} \\\\ 5 \\geq - \\mu + 10 - \\frac{9}{1 + \\mu}  \\iff \\mu^2  - 4\\mu + 4\\geq 0 \\iff (\\mu - 2)^2 \\geq 0$$\n",
        "Последнее выполнено всегда. Итоговый вид двойственной функции в пункте ниже"
      ],
      "metadata": {
        "id": "9LaMJgdbyF2m"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "x = np.linspace(-5, 5)\n",
        "plt.plot(x, x**2 + 1, label='$x^2 + 1$')\n",
        "plt.axvline(2,linestyle='--', color='red', label='Левая граница допустимых значений')\n",
        "plt.axvline(4,linestyle='--', color='red', label='Правая граница допустимых значений')\n",
        "plt.scatter(2, 5, color='red', label = 'Оптимум')\n",
        "plt.plot(x, x**2 + 1 + 0.5*(x-2)*(x-4), label='$L(x, 0.5)$')\n",
        "plt.plot(x, x**2 + 1 + 1*(x-2)*(x-4), label='$L(x, 1)$')\n",
        "plt.plot(x, x**2 + 1 + 2*(x-2)*(x-4), label='$L(x, 2)$')\n",
        "\n",
        "plt.xlim(-5, 5)\n",
        "plt.ylim(0, 20)\n",
        "plt.xlabel('x')\n",
        "plt.ylabel('y')\n",
        "plt.legend()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 473
        },
        "id": "-Zrt6Lfoy6q6",
        "outputId": "02590346-59e3-4731-f8f2-c057eadead0b"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.legend.Legend at 0x783156172c50>"
            ]
          },
          "metadata": {},
          "execution_count": 2
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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A6E95Ys92bp1KyNGjKBx48ZYrVauuuoqgGIyXcjZs2fLbAS4e/du+vTpU0Qp7devXyibpywkSUKn0RGQA4z+02j279/Pxo0bAcUtMnr06JDVLWgBulxkWebRRx/lueeeCylNFzJx4kTee+89AoEAs2fPZuLEiSWO69u3LzExMbS85mru/OtfObp/f4nj4uPjmT17NrNmzaJ58+Y8/vjjF5Vz7ty5REVFkZiYSMuWLZk2bVqxYxe+dy68jsuWLeOaa66hQYMGWK1W7rzzTs6ePRtypbVv356NGzeSkZFR4vG3b9/O/v37sVqtoWMkJCTgcrmKuJi/++67InJMmlR67Zfp06czaNAg+vfvX+qYYvd5eZBlpQfYrl2iFUYYOJXnwuZeS+vjymefsO9grbsWQgGqTGy20pcLg41PnSp97A8/FB176FDxMRVAr9Wg0yo/dDmV0Brjr3/9K4sWLeKVV15hzZo1pKen06FDh2Im9scee4z09HTS0tIYMGAAo0ePDj0RjxgxguzsbN5//302bdrEpk2bAEJz6HQ6Pv30U7788kvMZjPR0dG88sorly17WWzYsIFx48YxfPhwvvvuO7Zt28ZTTz1V7Lxat25Nenp6aPnggw9KnO/cuXO8+OKLzJgx47yiYbCApIXAxRVRu93OkCFDiImJYe7cufz6668sWrQI4KLujIMHD9K0adNynPWlEcwGs8RZGDFiBB999BEnT57khx9+KGK1Clp2LpdPPvkEu91e5g/1n/70J77//nvmz59PUlISHTp0KHHcggUL2LZtG5/+97/sP3KEB//yl1LnXL16NVqtlqysLOzlcD3feOONbNu2ja+//ppNmzaFrlfhYxe+d1JSUkLbDh06xA033EDHjh356quv2Lp1a+ihIXi97733Xnr06EGzZs1CykthbDYb3bp1K3KM9PR0/vjjD+64447QuEGDBhXZPn369BLPZ9++fXzwwQe89tprpZ5zife5IOL5MW0ffsc5LG7wmfUY3epNarhUhAusMikUaxK2saVg0CluixyHl/pWY5lfVD6fjy1bttCzZ09ACTzNycnhiiuUBnnr1q3jnnvuYdSoUYDypVs4yDJI3bp1Q3VBpk2bRocOHcjIyCA+Pp69e/fy/vvvh4Jl165dW2z/ESNG8Omnn+L1evnnP//JW2+9xerVqy/9QwA2btzIXXfd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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "mu = np.linspace(0, 5)\n",
        "plt.plot(mu, -mu + 10 - 9/(1+mu))\n",
        "plt.xlim(0, 5)\n",
        "plt.xlabel('$\\mu$')\n",
        "plt.ylabel('$g(\\mu)$')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 468
        },
        "id": "tsQ9rK05QINR",
        "outputId": "077945ce-27cc-4eda-e61e-24c3049364bf"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0, 0.5, '$g(\\\\mu)$')"
            ]
          },
          "metadata": {},
          "execution_count": 21
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**3.** State the dual problem, and verify that it is a concave maximization problem."
      ],
      "metadata": {
        "id": "hxi8V1R4FcRn"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "В прошлом пункте вывели инфимум:\n",
        "\n",
        "$$\\begin{cases}g(\\mu) = \\inf_x L(x, \\mu) = -\\mu + 10 -\\frac{9}{1+\\mu} \\to \\max_{\\mu} \\\\ \\mu \\geq 0\\end{cases}$$\n",
        "\n",
        "Докажем вогнутость. Допустимое множество выпукло. Докажем выпуклость $-g(\\mu)$:\n",
        "$$\\frac{-g}{d\\mu d\\mu} = \\frac{18(1+\\mu)}{(1+\\mu)^3} \\geq 0$$\n",
        "По н.д.у. выпуклая на допустимом множестве. Следовательно изначальная функция/задача вогнута."
      ],
      "metadata": {
        "id": "jMIp6YAS68nX"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**4.** Find the dual optimal value and dual optimal solution $\\mu^*$. Does strong duality hold?\n",
        "Let $p^*(u)$ denote the optimal value of the problem\n",
        "\n",
        "$$\n",
        "\\begin{split}\n",
        "& x^2 + 1 \\to \\min\\limits_{x \\in \\mathbb{R} }\\\\\n",
        "\\text{s.t. } & (x-2)(x-4) \\leq u\n",
        "\\end{split}\n",
        "$$\n",
        "as a function of the parameter $u$. Plot $p^*(u)$. Verify that $\\dfrac{dp^*(0)}{du} = -\\mu^*$"
      ],
      "metadata": {
        "id": "cbhGvftmFdth"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Запишем лагранжиан:\n",
        "$$L(x, \\mu) = x^2 + 1 + \\mu(x^2 - 6x + 8 - u)$$\n",
        "Найдем инфинум и запишем двойственную задачу:\n",
        "$$ \\inf_x L(x, \\mu): \\\\\n",
        "\\begin{cases} \\frac{dL}{dx} = 2x + 2\\mu x - 6\\mu = 0 \\\\ \\frac{dL}{dxdx} = 2(1+\\mu) > 0 \\end{cases} \\to x_{inf} = \\frac{3\\mu}{1+\\mu} \\to \\\\\n",
        "g(\\mu) = \\inf_x L(x, \\mu) = \\frac{9\\mu^2 - 18\\mu^2}{1+\\mu} + 1 + (8 - u)\\mu = -(1+u)\\mu + 10 - \\frac{9}{1+\\mu} \\to \\\\ \\begin{cases} g(\\mu) = -(1+u)\\mu + 10 - \\frac{9}{1+\\mu} \\to \\max_\\mu \\\\ \\mu \\geq 0\\end{cases}$$\n",
        "Решим задачу (изначально знаем, что задача вогнутая):\n",
        "$$u > -1 : \\frac{dg}{d\\mu} = -1 -u + \\frac{9}{(1+\\mu)^2} = 0 \\to \\mu = \\sqrt{\\frac{9}{1+u}} - 1 \\\\ u \\leq -1:\\frac{dg}{d\\mu} > 0 \\to \\mu = +∞ $$\n",
        "\n",
        "Последний пункт верен, так как, если это возрастающая функция на выпуклом множестве (луче), то её макcимум \"находится\" на бесконечности. Сразу отметим, что $u < -1$ означает, что допустимое множество изначальной функции пустое - $(x-2)(x-4) \\geq -1$. Сразу отметем этот вариант, так как тут нет задачи, чтобы проверить двойственность. Для остальных проверим сильную двойственность. Найдем оптимальное значение двойственной функции:\n",
        "$$g^* = \\begin{cases}11 -6\\sqrt{1+u} + u| u > -1 \\\\\n",
        "10 | u = -1 \\end{cases}$$\n",
        "\n",
        "Теперь же решим изначальную задачу:\n",
        "\n",
        "Сразу разберем случай $u=-1$ - допустимое множество - единственная точка $x=3 \\to y^* = 9 + 1 = 10$ - сильная двойственность есть.\n",
        "\n",
        "Разберем $u>-1$:\n",
        "\n",
        "В таком случае, возьмём точку $x_0 = 3$:\n",
        "1. $(x_0-2)(x_0-4) = -1 < u$ - строго допустимая точка\n",
        "2. $(x-2)(x-4) - u$ - это выпуклая функция (паробола с ветвями вверх)\n",
        "3. $x^2 +1$ - аналогично\n",
        "4. Можем добавить воображаемое афинное ограничение: $0x = 0$, тогда $h(x_0) = 0$\n",
        "\n",
        "В таком случае выполняются условия Слейтера и сильная двойственность существует. В таком случае:\n",
        "\n",
        "$$p^*(u) = g^*(u) = \\begin{cases}11 -6\\sqrt{1+u} + u | u > -1 \\\\\n",
        "10 | u = -1 \\end{cases}$$\n",
        "\n",
        "Проверим то, что просят в задаче:\n",
        "\n",
        "$$\\frac{dp^*(0)}{du} =  1 - 3\\frac{1}{(1+u)^{0.5}} |_{u=0} = -2\\\\ \\mu^* = \\sqrt{\\frac{9}{1+u}} - 1 = 3 - 1 = 2$$\n",
        "\n",
        "Выполнено, построим график:"
      ],
      "metadata": {
        "id": "8bvYmT1O-HAr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "u = np.linspace(-1, 5)\n",
        "plt.plot(u, 11 - 6*np.sqrt(1+u) + u)\n",
        "\n",
        "plt.xlim(-1, 5)\n",
        "plt.xlabel('u')\n",
        "plt.ylabel('$p^*(u)$')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 467
        },
        "id": "vTnxNBHcO3EQ",
        "outputId": "14dd6185-f50f-44fb-b6e5-935f9bf0f3ae"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0, 0.5, '$p^*(u)$')"
            ]
          },
          "metadata": {},
          "execution_count": 19
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 2"
      ],
      "metadata": {
        "id": "20Lqn-RHQpRS"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Consider a smooth convex function $f(x)$ at some point $x_k$. One can define the first-order Taylor expansion of the function as: $$\n",
        "f^I_{x_k}(x) = f(x_k) + \\nabla f(x_k)^\\top (x - x_k),\n",
        "$$ where we can define $\\delta x = x - x_k$ and $g = \\nabla f(x_k)$. Thus, the expansion can be rewritten as: $$\n",
        "f^I_{x_k}(\\delta x) = f(x_k) + g^\\top \\delta x.\n",
        "$$ Suppose, we would like to design the family of optimization methods that will be defined as: $$\n",
        "x_{k+1} = \\text{arg}\\min_{\\delta x} \\left\\{f^I_{x_k}(\\delta x) + \\frac{\\lambda}{2} \\|\\delta x\\|^2\\right\\},\n",
        "$$ where $\\lambda > 0$ is a parameter.\n",
        "\n"
      ],
      "metadata": {
        "id": "qRs9H-lxRzCw"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "[5 points] Show, that this method is equivalent to the gradient descent method with the choice of Euclidean norm of the vector $\\|\\delta x\\| = \\|\\delta x\\|_2$. Find the corresponding learning rate."
      ],
      "metadata": {
        "id": "dwynWMg4R46M"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "$$x_{k+1} = \\text{arg}\\min_{\\delta x} \\left\\{f^I_{x_k}(\\delta x) + \\frac{\\lambda}{2} \\|\\delta x\\|_2^2\\right\\} = \\text{arg}\\min_{\\delta x} \\left\\{f(x_k) + g^Tδx + \\frac{\\lambda}{2} \\|\\delta x\\|_2^2\\right\\} $$\n",
        "Заметим, что функция выпукла. Действительно, при $\\lambda > 0: \\frac{\\lambda}{2}||δx||_2^2$ - выпуклая функция (из лекции про выпуклость). $f(x_k) $- константа, $g^Tδx$ - линейная функция - они тоже выпуклы (гессиан - нулевая матрица, следовательно положительно полуопределен). Сумма выпуклых функций - выпукла. Найдем минимум:\n",
        "$$\\nabla_{\\delta x}\\left[ f^I_{x_k}(\\delta x) + \\frac{\\lambda}{2} \\|\\delta x\\|_2^2 \\right] = g + \\lambda \\delta x = 0 \\to \\delta x_{\\min} = -\\frac{g}{\\lambda}\\\\ \\delta x_{\\min} = x_{k+1} - x_k =  -\\frac{g}{\\lambda} \\to x_{k+1} = x_k - \\frac{g}{\\lambda}$$\n",
        "Действительно получили выражение для градиентного спуска с рейтом $\\frac{1}{\\lambda}$.\n"
      ],
      "metadata": {
        "id": "0kwYk4rxS59x"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "[5 points] Prove, that the following holds: $$\n",
        "\\text{arg}\\min_{\\delta x \\in \\mathbb{R}^n} \\left\\{ g^T\\delta x + \\frac{\\lambda}{2} \\|\\delta x\\|^2\\right\\} = - \\frac{\\|g\\|_*}{\\lambda} \\text{arg}\\max_{\\|t\\|=1} \\left\\{ t^T g \\right\\},\n",
        "$$ where $\\|g\\|_*$ is the dual norm of $g$."
      ],
      "metadata": {
        "id": "7jOE4iy7R5aD"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Сделаем замену:\n",
        "$$\\delta x = \\alpha t: \\alpha \\in \\mathbb{R_+}, t \\in \\mathbb{R}^n , ||t|| = 1$$\n",
        "\n",
        "Тогда можем, преобразовать задачу:\n",
        "\n",
        "$$\\text{arg}\\min_{\\alpha, t \\to \\delta x = \\alpha t} \\left\\{ \\alpha g^T t + \\frac{\\lambda}{2} \\alpha^2 \\right\\}$$\n",
        "\n",
        "Эта задача эквивалентна, так как каждой комбинации $(\\alpha, t)$ соотвествует один x. И наоборот за исключением, $\\alpha = 0$. Кроме того функция выпукла по обоим аргументам. Покажем это:\n",
        "$$\\nabla^2_{(\\alpha, t)}f = \\left(\\begin{matrix}\n",
        "\\lambda & g^t \\\\\n",
        "g & 0  \\\\\n",
        "\\end{matrix}\\right)$$\n",
        "Первый минор - положительный. Любой другой будет $$M_k = ||(g_1, g_2, \\dots, g_{k-1})||_2^2 \\geq 0$$\n",
        "Нормой обрезанного вектора градиента. По н.д.у функция выпукла.\n",
        "\n",
        "Отлично, теперь прооптимизируем:\n",
        "\n",
        "$$\\alpha: \\\\\n",
        "\\text{Парабола, оптимум - вершина: }\\alpha^* = -\\frac{g^T t}{\\lambda} \\to \\hat{f}(t) =  -\\frac{g^T t}{\\lambda} g^T t + \\frac{\\lambda}{2} \\frac{g^T t g^T t}{\\lambda} = \\\\ = -\\frac{(g^T t)^2}{2\\lambda} \\\\\n",
        "t: \\\\\n",
        "t^* = \\text{arg}\\min_{||t|| = 1}  -\\frac{(g^T t)^2}{2\\lambda}  = \\text{arg}\\max_{||t|| = 1} \\frac{(g^T t)^2}{2\\lambda} = \\text{arg}\\max_{||t|| = 1} (g^T t)^2 \\overset{\\text{Пусть } g^T t > 0}{=}  \\text{arg}\\max_{||t|| = 1} <g,t> $$\n",
        "\n",
        "Тогда:\n",
        "$$\\alpha^* =-\\frac{(g^T t^*)}{\\lambda} = -\\frac{<g, t^*>}{\\lambda} = -\\frac{<g, \\text{arg}\\max_{||t|| = 1} <g,t> >}{\\lambda} = -\\frac{\\sup_{||t||=1}<g, t>}{\\lambda} = -\\frac{||g||^*}{\\lambda}$$\n",
        "\n",
        "Итого:\n",
        "$$\\delta x^* = \\alpha^* t^* = -\\frac{||g||^*}{\\lambda} \\text{arg}\\max_{||t|| = 1} <g,t> = -\\frac{||g||^*}{\\lambda} \\text{arg}\\max_{||t|| = 1} (t^T g)$$"
      ],
      "metadata": {
        "id": "NX-ILe4tS6c4"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "[3 points] Consider another vector norm $\\|\\delta x\\| = \\|\\delta x\\|_\\infty$. Write down exact expression for the corresponding method."
      ],
      "metadata": {
        "id": "3kOt-o1bR5hQ"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Покажем, что: $||\\delta x||^*_{\\infty} = ||\\delta x||_1$:\n",
        "$$||\\delta x||^*_{\\infty}  = \\sup_{||x||_{\\infty}\\leq 1} <\\delta x, x> = \\sup_{\\max(\\{|x_1|, \\dots, |x_n|\\})\\leq 1} (x_1\\delta x_1+ \\dots+ x_n\\delta x_n) = |\\delta x_1| + \\dots + |\\delta x_n| = ||\\delta x||_1$$\n",
        "\n",
        "Тогда:\n",
        "$$x_{k+1} =\\text{arg}\\min_{\\delta x} \\left\\{f(x_k) + g^Tδx + \\frac{\\lambda}{2} \\|\\delta x\\||_{\\infty}^2\\right\\} = \\text{arg}\\min_{\\delta x} \\left\\{g^Tδx + \\frac{\\lambda}{2} \\|\\delta x\\|_{\\infty}^2\\right\\} = - \\frac{\\|g\\|_1}{\\lambda} \\text{arg}\\max_{\\|t\\|_{\\infty}=1} \\left\\{ t^T g \\right\\} =  - \\frac{\\|g\\|_1}{\\lambda} \\text{arg}\\max_{\\max(\\{|t_1|, \\dots, |t_n|\\})= 1} \\left\\{ t_1 g_1 + \\dots + t_n g_n\\right\\} =  - \\frac{\\|g\\|_1}{\\lambda}\\left(\\begin{matrix}\n",
        "\\frac{g_1}{|g_1|} \\\\\n",
        "\\frac{g_2}{|g_2|}  \\\\\n",
        "\\vdots\\\\\n",
        "\\frac{g_n}{|g_n|}\n",
        "\\end{matrix}\\right)$$"
      ],
      "metadata": {
        "id": "OLg-OkQGS68H"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "[2 points] Consider induced operator matrix norm for any matrix $W \\in \\mathbb{R}^{d_{out} \\times d_{in}}$ $$\n",
        "\\|W\\|_{\\alpha \\to \\beta} = \\max_{x \\in \\mathbb{R}^{d_{in}}} \\frac{\\|Wx\\|_{\\beta}}{\\|x\\|_{\\alpha}}.\n",
        "$$ Typically, when we solve optimization problems in deep learning, we stack the weight matrices for all layers $l = [1, L]$ into a single vector. $$\n",
        "w = \\text{vec}(W_1, W_2, \\ldots, W_L) \\in \\mathbb{R}^{n},\n",
        "$$ Can you write down the exact expression, that relates $$\n",
        "\\|w\\|_\\infty \\qquad \\text{ and } \\qquad \\|W_l\\|_{\\alpha \\to \\beta}, \\; l = [1, L]?\n",
        "$$"
      ],
      "metadata": {
        "id": "xC1ey3-7R5oR"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "$$||w||_\\infty = \\max(|w^{(1)}_{11}|, \\dots, |w^{(1)}_{d_{out}d_{in}}|, \\dots, |w^{(L)}_{11}|, \\dots, |w^{(L)}_{d_{out}d_{in}}|)$$\n",
        "Нам нужно получить максимальный элемент среди всех матриц. Попробуем извлечь максимальный элемент из матрицы посредством указанной нормы:\n",
        "$$||W||_{\\alpha \\to \\beta} = \\sup_{x\\in \\mathbb{R}^{d_{in}}} \\frac{||Wx||_\\beta}{||x||_{\\alpha}}$$\n",
        "Попробуем извлечь элемент, взяв $\\alpha = 1, \\beta = \\infty$:\n",
        "$$||W||_{1 \\to \\infty} = \\sup_{x\\in \\mathbb{R}^{d_{in}}} \\frac{||Wx||_\\infty}{||x||_{1}} =  \\sup_{x\\in \\mathbb{R}^{d_{in}}} \\frac{\\max\\{<w^{(1)}, x>, \\dots, <w^{(d_{out})}, x>\\}}{|x_1| + \\dots + |x_{d_{in}}|} = \\sup_{||x||_1 = 1} \\max\\{<w^{(1)}, x>, \\dots, <w^{(d_{out})}, x>\\} = \\max\\{\\sup_{||x||_1 = 1} <w^{(1)}, x>, \\dots, \\sup_{||x||_1 = 1} <w^{(d_{out})}, x>\\} \\xrightarrow{\\text{dual norm expression for } ||\\dots||_1} = \\max\\{ ||w^{(1)}||_\\infty, \\dots, ||w^{(d_{out})}||_\\infty\\} = \\max_{i,j} W_{i. j}$$\n",
        "Отлично и тогда можем записать:\n",
        "$$||w||_\\infty = \\max_{l\\in[1, L]} ||W_i||_{1 \\to \\infty} $$"
      ],
      "metadata": {
        "id": "0YeIhBDMS7cb"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 3"
      ],
      "metadata": {
        "id": "tzZf6aReWTMq"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Derive the dual problem for the Ridge regression problem with $A \\in \\mathbb{R}^{m \\times n}, b \\in \\mathbb{R}^m, \\lambda > 0$:\n",
        "$$\n",
        "\\begin{split}\n",
        "\\dfrac{1}{2}\\|y-b\\|^2 + \\dfrac{\\lambda}{2}\\|x\\|^2 &\\to \\min\\limits_{x \\in \\mathbb{R}^n, y \\in \\mathbb{R}^m }\\\\\n",
        "\\text{s.t. } & y = Ax\n",
        "\\end{split}\n",
        "$$"
      ],
      "metadata": {
        "id": "WTXxQ1fsYnnz"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Выпишем Лагранжиан:\n",
        "$$L(x,y, \\nu) = \\dfrac{1}{2}\\|y-b\\|^2 + \\dfrac{\\lambda}{2}\\|x\\|^2 + \\nu^T(Ax - y)$$\n",
        "Найдем его инфимум по $(x,y)$ (задача выпукла, так как вторые нормы и афинное преобразование):\n",
        "$$\\nabla_x L = \\lambda x + A^T \\nu = 0 \\to x = -\\frac{A^T \\nu}{\\lambda}\\\\\n",
        "\\nabla_y L = (y-b) - \\nu = 0 \\to y = \\nu + b \\to \\\\\n",
        "\\inf_{x,y} L(x,y, \\nu) = \\dfrac{1}{2}\\|\\nu\\|^2 + \\dfrac{\\lambda}{2}\\|-\\frac{A^T \\nu}{\\lambda}\\|^2 + \\nu^T(-A\\frac{A^T \\nu}{\\lambda} - \\nu - b) = \\dfrac{1}{2}\\|\\nu\\|^2 + \\dfrac{1}{2\\lambda}\\|A^T \\nu\\|^2 - (\\frac{\\nu^TAA^T \\nu}{\\lambda} + \\nu^T\\nu + \\nu^Tb)  = \\dfrac{1}{2}\\|\\nu\\|^2 + \\dfrac{1}{2\\lambda}\\|A^T \\nu\\|^2 - (\\frac{||A^T \\nu||}{\\lambda} + ||\\nu||^2 + <\\nu, b>) =   -\\dfrac{1}{2}\\|\\nu\\|^2 - \\dfrac{1}{2\\lambda}\\|A^T \\nu\\|^2 -<\\nu, b> $$\n",
        "\n",
        "Запишем двойственную задачу:\n",
        "$$g(\\nu) = -\\dfrac{1}{2}\\|\\nu\\|^2 - \\dfrac{1}{2\\lambda}\\|A^T \\nu\\|^2 -<\\nu, b> \\to \\max_{\\nu \\in \\mathbb{R^m}}$$\n"
      ],
      "metadata": {
        "id": "XAshSGEUYoVU"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 4"
      ],
      "metadata": {
        "id": "Npw6nY-2dSQx"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Derive the dual problem for the support vector machine problem with $A \\in \\mathbb{R}^{m \\times n}, \\mathbf{1} \\in \\mathbb{R}^m \\in \\mathbb{R}^m, \\lambda > 0$:\n",
        "$$\n",
        "\\begin{split}\n",
        "\\langle \\mathbf{1}, t\\rangle + \\dfrac{\\lambda}{2}\\|x\\|^2 &\\to \\min\\limits_{x \\in \\mathbb{R}^n, t \\in \\mathbb{R}^m }\\\\\n",
        "\\text{s.t. } & Ax \\succeq \\mathbf{1} - t \\\\\n",
        "& t \\succeq 0\n",
        "\\end{split}\n",
        "$$"
      ],
      "metadata": {
        "id": "ED5c8FvwdTro"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Запишем Лагранжиан:\n",
        "$$L(x, t, \\mu, \\nu) = \\langle \\mathbf{1}, t\\rangle + \\dfrac{\\lambda}{2}||x||^2 + \\mu^T(1-t - Ax) - \\nu^T t$$\n",
        "Инфимум (задача аналогично выпуклая - сумма линейной функции, квадрата нормы и афинного преобразования - выпукло):\n",
        "$$\\nabla_x L = \\lambda x - A^T \\mu = 0 \\to x = \\frac{A^T \\mu}{\\lambda} \\\\\n",
        "t : L(x, t, \\mu, \\nu) =  \\langle \\mathbf{1}, t\\rangle + \\dfrac{\\lambda}{2}||x||^2 + \\mu^T(1-t - Ax) - \\nu^T t =  \\langle \\mathbf{1} - \\mu - \\nu, t\\rangle + \\dfrac{\\lambda}{2}||x||^2 + \\mu^T(1 - Ax)$$\n",
        "Как видим, это линейная функция по t. Если $\\mathbf{1} - \\mu - \\nu \\succeq 0$, то оптимальный $t = 0$. Иначе $t = \\infty$, нет нижней границы. Значит используем это в качестве ограничения. Выпишем инфимум и двойственную задачу:\n",
        "$$\\inf_{x,t}L(x, t, \\mu, \\nu) =  \\dfrac{\\lambda}{2}||\\frac{A^T \\mu}{\\lambda}||^2 + \\mu^T(1 - A\\frac{A^T \\mu}{\\lambda}) = -\\frac{\\lambda}{2}||\\frac{A^T \\mu}{\\lambda}||^2 +  \\langle \\mathbf{1}, \\mu\\rangle\\\\\n",
        "\\begin{cases}g(\\mu, \\nu) = -\\frac{\\lambda}{2}||\\frac{A^T \\mu}{\\lambda}||^2 +  \\langle \\mathbf{1}, \\mu\\rangle \\\\ \\mu \\geq 0 \\\\ \\nu \\geq 0 \\\\ \\mathbf{1} - \\mu - \\nu \\succeq 0\\end{cases} ≡\\begin{cases}g(\\mu) = -\\frac{\\lambda}{2}||\\frac{A^T \\mu}{\\lambda}||^2 +  \\langle \\mathbf{1}, \\mu\\rangle \\\\ \\mathbf{1} ≽ \\mu ≽ 0 \\end{cases} $$"
      ],
      "metadata": {
        "id": "J3koZYkodXx4"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 5"
      ],
      "metadata": {
        "id": "1A0IpJoTlAwr"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Give an explicit solution to the following LP.\n",
        "$$\n",
        "\\begin{split}\n",
        "& c^\\top x \\to \\min\\limits_{x \\in \\mathbb{R}^n }\\\\\n",
        "\\text{s.t. } & 1^\\top x = 1, \\\\\n",
        "& x \\succeq 0\n",
        "\\end{split}\n",
        "$$\n",
        "This problem can be considered the simplest portfolio optimization problem."
      ],
      "metadata": {
        "id": "8WHo6wEnlCOM"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Заметим, что выполняются условия Слейтера:\n",
        "1. $c^Tx$ - выпуклая функция (линейная)\n",
        "2. Точка $x = (\\frac{1}{n}, \\dots, \\frac{1}{n},)$ - строго допустимая: $1^Tx = n\\frac{1}{n} = 1$, $\\forall i: x_i = \\frac{1}{n} > 0$\n",
        "\n",
        "В таком случае есть сильная двойственность. Выпишем Лагранжиан и двойственную задача:\n",
        "$$L(x, \\lambda, \\nu) = c^T x - \\lambda^T x + \\nu(1^Tx - 1) = \\langle x, c - \\lambda + \\nu 1 \\rangle - \\nu$$\n",
        "При $c - \\lambda + \\nu 1 ≽ 0$ инфимум $\\inf L = -\\nu$ (при $x≽0$) достигается при $x=0$, иначе нижней границы нет. Двойственная задача:\n",
        "$$\\begin{cases} -\\nu \\to \\max ≡ \\nu \\to \\min\\\\\n",
        "\\lambda ≽ 0 \\\\\n",
        "c - \\lambda + \\nu 1 ≽ 0 \\to \\nu 1≽\\lambda - c ≽ -c\\end{cases} \\to \\nu = \\max_{i}(-c_i) = \\min(c_i) = ||-c||_\\infty \\to c^tx^* = -||-c||_\\infty$$\n",
        "\n",
        "Остается сконструировать подходящий x. Подходящим будет, например, $$\\forall i \\neq \\text{arg}\\min c_i: x_i = 0, x_{\\text{arg}\\min c_i} = 1$$\n",
        "Оба условия выполняются, а $c^t x = \\min(c_i) = -||-c||_\\infty$ по построению."
      ],
      "metadata": {
        "id": "ATzxkeqKlRUk"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 6"
      ],
      "metadata": {
        "id": "pSPYqlBEyN5v"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Show, that the following problem has a unique solution and find it:\n",
        "$$\n",
        "\\begin{split}\n",
        "& \\langle C^{-1}, X\\rangle - \\log \\det X \\to \\min\\limits_{x \\in \\mathbb{R}^{n \\times n} }\\\\\n",
        "\\text{s.t. } & \\langle Xa, a\\rangle \\leq 1,\n",
        "\\end{split}\n",
        "$$\n",
        "where $C \\in \\mathbb{S}^n_{++}, a \\in \\mathbb{R}^n \\neq 0$. The answer should not involve inversion of the matrix $C$."
      ],
      "metadata": {
        "id": "jmsvse_LyUsA"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Во-первых, оптимизируемая функция - выпуклая. Так как:\n",
        "1. $<C^{-1}, X>$ - линейная - выпуклая\n",
        "2. $-\\log \\det X$ - выпуклая (из лекции), покажем, что она строго выпукла\n",
        "\n",
        "$$\\nabla_X -\\log \\det X = -\\frac{\\det X X^{-T}}{\\det X} = -X^{-T}\\\\\n",
        "\\delta \\nabla_X-\\log \\det X = X^{-T}\\delta X^T X^{-T} ≻ 0$$\n",
        "\n",
        "Последнее выполнено в силу положительной определенности $X^{-1}$, т.к. $X \\in S_{++}$\n",
        "\n",
        "Сумма выпуклой и строго выпуклой функции дает нам строго выпуклую функцию. Следовательно единственное решение на выпуклом множестве.\n",
        "\n",
        "Докажем, что множество выпукло:\n",
        "$$\\{X \\in S_{++}:\\langle Xa, a \\rangle\\}\\leq 1 \\} \\equiv \\{X \\in S_{++}:a^t X a\\leq 1 \\} \\\\\n",
        "X_1, X_2 \\in \\{X \\in S_{++}:a^t X a\\leq 1 \\}, \\theta \\in [0,1]\\to \\\\\n",
        "a^T(\\theta X_1 + (1-\\theta)X_2)a = \\theta  a^TX_1 a + (1-\\theta)a^TX_2 a\\\\\n",
        "\\theta  \\overset{> 0}{a^TX_1 a} + (1-\\theta)\\overset{>0 }{a^TX_2 a} \\to \\theta X_1 + (1-\\theta)X_2\\in S_{++} \\\\\n",
        "\\theta  \\overset{\\le 1}{a^TX_1 a} + (1-\\theta)\\overset{\\le 1}{a^TX_2 a} \\le \\theta + 1 - \\theta = 1\\to \\theta X_1 + (1-\\theta)X_2\\in \\{X \\in S_{++}:a^t X a\\leq 1 \\} $$\n",
        "\n",
        "Показали единственность решения. Найдем его.\n",
        "\n",
        "Из пунктов выше показали, выпуклость функций. Покажем, что существут строго доступная точка и условия Слейтера выполнены:\n",
        "\n",
        "$$0 < \\langle Xa, a \\rangle \\leq 1 \\to 0 < \\langle \\frac{X}{2}a, a \\rangle < 1$$\n",
        "\n",
        "Так как множество выпуклое, то поделенная на 2 точка ($\\theta = 0.5$) все еще дает точку из множества. Условия Слейтера выполнены, ККТ - необходимы и достаточны. Выпишем Лагранжиан и ККТ.\n",
        "\n",
        "$$L(X, \\lambda) = \\langle C^{-1}, X\\rangle-\\log \\det X + \\lambda(\\langle Xa, a \\rangle - 1)\\\\\n",
        "\\nabla_X L = C^{-1} - X^{-T} + \\lambda a a^T =  C^{-1} - X^{-1} + \\lambda a a^T = 0 \\to X^{-1} = C^{-1}+ \\lambda a a^T \\xrightarrow{\\text{Шерман-Моррисон из курса линала}} X = C - \\frac{\\lambda Caa^TC}{1+\\lambda a^TCa} \\\\ \\lambda \\geq 0 \\\\ \\lambda (\\langle Xa, a \\rangle - 1) = 0 \\\\\n",
        "\\langle Xa, a \\rangle - 1 \\leq 0$$\n",
        "Разеберем последнее:\n",
        "$$\\langle X^*a, a \\rangle - 1 = \\langle Ca, a \\rangle - \\langle \\frac{\\lambda Caa^TC}{1+\\lambda a^TCa}a, a \\rangle   - 1 = a^T C a - \\frac{\\lambda a^tCaa^TCa}{1+\\lambda a^TCa} - 1 = a^t C a -  \\frac{\\lambda(a^TCa)^2}{1+\\lambda a^TCa} - 1$$\n",
        "Ограничение неактивно, если:\n",
        "$$\\langle X^*a, a \\rangle - 1 = a^t C a -  \\frac{\\lambda(a^TCa)^2}{1+\\lambda a^TCa} - 1 < 0\\to x-  \\frac{\\lambda x^2}{1+\\lambda x} - 1 < 0 \\to \\frac{x + \\lambda x^2 - \\lambda x^2 - 1 - \\lambda x}{1+\\lambda x} = \\frac{x - 1 - \\lambda x}{\\overset{>0}{1+\\lambda x}} < 0 \\to x - 1 - \\lambda x < 0 \\to x <\\frac{1}{1-\\lambda}$$\n",
        "При неактивном ограничении: $\\lambda = 0 \\to a^t C a < 1$. И тогда $X^* = C- 0 = C$.\n",
        "\n",
        "Пусть ограничение активно: $\\lambda > 0$. Из предыдущего, при активном ограничении:\n",
        "$$x - 1 - \\lambda x = 0 \\to \\lambda = \\frac{x-1}{x} = \\frac{a^t C a - 1}{a^t C a }$$ Тогда:\n",
        "$$X^* =  C - \\frac{\\lambda Caa^TC}{1+\\lambda a^TCa} =  C - \\frac{a^t C a - 1}{a^t C a }\\frac{ Caa^TC}{1+\\frac{a^t C a - 1}{a^t C a } a^TCa} = C -\\frac{a^t C a - 1}{a^t C a }\\frac{ Caa^TC}{1+a^t C a - 1 } =   C -\\frac{a^t C a - 1}{a^t C a }\\frac{ Caa^TC}{a^t C a  } = C -\\frac{ a^t C aCaa^TC - Caa^TC}{(a^t C a)^2 }$$\n",
        "\n",
        "Итого:\n",
        "$$X^* = \\begin{cases}C \\text{ если } a^TCa < 1 \\\\ C -\\frac{ a^t C aCaa^TC - Caa^TC}{(a^t C a)^2 }\\end{cases}$$"
      ],
      "metadata": {
        "id": "UyG3NTwGyU--"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 7"
      ],
      "metadata": {
        "id": "ehSGgmonDgoO"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Give an explicit solution to the following QP.\n",
        "$$\n",
        "\\begin{split}\n",
        "& c^\\top x \\to \\min\\limits_{x \\in \\mathbb{R}^n }\\\\\n",
        "\\text{s.t. } & (x - x_c)^\\top A (x - x_c) \\leq 1,\n",
        "\\end{split}\n",
        "$$\n",
        "where $A \\in \\mathbb{S}^n_{++}, c \\neq 0, x_c \\in \\mathbb{R}^n$."
      ],
      "metadata": {
        "id": "HZmXTrRnDr5k"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Условия Слейтера выполняются. Функции выпуклы. Первую рзазобрали ранее. Вторая:\n",
        "$$(x - x_c)^\\top A (x - x_c) - 1 = ||x-x_c||_A^2 - 1$$\n",
        "Квадрат любой нормы - выпуклая, константа на выпуклость не влияет, сдвиг тоже. Покажем строго доступную точку: $x = x_c \\to  ||x-x_c||_A^2 - 1 = -1< 0$.\n",
        "\n",
        "Значит можем пользоваться ККТ, как необходимыми и достаточными условиями. Пишем ККТ и Лагранжиан:\n",
        "$$L(x, \\lambda) = c^Tx + \\lambda(x - x_c)^\\top A (x - x_c) - \\lambda \\\\\n",
        "\\nabla_x L = c +2\\lambda A(x-x_c) = 0 \\to Ax = \\frac{2\\lambda A x_c - c}{2\\lambda} \\xrightarrow{A\\in S_{++} \\to |A| > 0} x =  x_c - \\frac{A^{-1}c}{2\\lambda} \\\\\n",
        "\\lambda \\geq 0 \\\\\n",
        "\\lambda(x - x_c)^\\top A (x - x_c) - \\lambda = 0 \\\\\n",
        " (x - x_c)^\\top A (x - x_c) - 1 \\le 0$$\n",
        "\n",
        "Мне порядком надоело столько техать, внезапная замена: $y = x- x_c$\n",
        "$$L(y, \\lambda) =  c^T(y+x_c) + \\lambda y^\\top A y - \\lambda \\\\ y^*  = -\\frac{A^{-1}c}{2\\lambda}\\\\\n",
        "\\lambda \\geq 0 \\\\\n",
        "\\lambda y^\\top Ay - \\lambda = 0 \\\\\n",
        " y^\\top A y - 1 \\le 0$$\n",
        "\n",
        "Ограничения неактивны: $\\lambda = 0$. Тогда у нас возникает линейная функция по y, и нижняя граница исчезает. В таком случае оптимум будет на бесконечности, что не попадает под ограничения. Следовательно ограничения активны всегда.\n",
        "\n",
        "Ограничения активны: $\\lambda>0$:\n",
        "$$y^\\top A y - 1 = 0 \\to \\frac{c^TA^{-1}}{2\\lambda}A\\frac{A^{-1}c}{2\\lambda}= 1 \\to \\frac{c^TA^{-1}c}{4\\lambda^2} = 1 \\to \\lambda = \\frac{\\sqrt{c^TA^{-1}c}}{2} \\\\\n",
        "y^* =  -\\frac{A^{-1}c}{2\\lambda} =  -\\frac{A^{-1}c}{2 \\frac{\\sqrt{c^TA^{-1}c}}{2}}=-\\frac{A^{-1}c}{\\sqrt{c^TA^{-1}c}}$$\n",
        "Итого:\n",
        "$$x^* = y^* + x_c = x_c - \\frac{A^{-1}c}{\\sqrt{c^TA^{-1}c}}$$\n",
        "\n"
      ],
      "metadata": {
        "id": "ve92HHOUDuWI"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 8"
      ],
      "metadata": {
        "id": "x8pwVtJ9N9SE"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Consider the equality-constrained least-squares problem\n",
        "$$\n",
        "\\begin{split}\n",
        "& \\|Ax - b\\|_2^2 \\to \\min\\limits_{x \\in \\mathbb{R}^n }\\\\\n",
        "\\text{s.t. } & Cx = d,\n",
        "\\end{split}\n",
        "$$\n",
        "where $A \\in \\mathbb{R}^{m \\times n}$ with $\\mathbf{rank }A = n$, and $C \\in \\mathbb{R}^{k \\times n}$ with $\\mathbf{rank }C = k$. Give the KKT conditions, and derive expressions for the primal solution $x^*$ and the dual solution $\\lambda^*$."
      ],
      "metadata": {
        "id": "kc0rj8fqOHkX"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Сначала укажем корректность использования ККТ. Условия Слейтера выполняются, так как, да да не удивляйтесь, $||Ax -b||^2_2$ - выпуклая, как выпуклая (квадрат нормы) от афинного преобразования. $Cx = d$ - афинное ограничение. Наличие строго допустимой точки напрямую зависит от наличия допустимых точек вообще. $Cx = d$, если решается, то решение - строго допустимая точка.\n",
        "\n",
        "Тем самым ККТ необходимы и достаточно. Лагранжиан+ККТ:\n",
        "$$L(x, \\lambda) = ||Ax -b ||_2^2 + \\lambda^T(Cx - d) \\\\\n",
        "\\nabla_x L = 2A^T(Ax-b) +  C^T \\lambda = 0 \\to A^TAx =A^Tb -\\dfrac{1}{2} C^T \\lambda\\xrightarrow{\\text{rank}(A^TA) = n} x^* = (A^TA)^{-1}(A^Tb -\\dfrac{1}{2} C^T \\lambda)\\\\\n",
        "\\nabla_{\\lambda} L = Cx-d = 0 \\to C(A^TA)^{-1}(A^Tb -\\dfrac{1}{2} C^T \\lambda) = d \\to  C(A^TA)^{-1}(C^T \\lambda) = 2C(A^TA)^{-1}(A^Tb) - 2d$$\n",
        "Заметим, что C-полноранговая по строкам следовательно $C(A^TA)^{-1}$ тоже, следовательно и $C(A^TA)^{-1}C^T \\in \\mathbb{R}^{k\\times k}$.\n",
        "$$(C(A^TA)^{-1}C^T) \\lambda^* = 2C(A^TA)^{-1}(A^Tb) - 2d \\to \\lambda^* = 2(C(A^TA)^{-1}C^T)^{-1}(C(A^TA)^{-1}(A^Tb) - d)$$\n",
        "Больше условий нет, так как ограничение вида равенство.\n",
        "\n",
        "Тогда итоговые решения:\n",
        "$$\\lambda^* = 2(C(A^TA)^{-1}C^T)^{-1}(C(A^TA)^{-1}(A^Tb) - d) \\\\\n",
        "x^* = (A^TA)^{-1}(A^Tb - C^T (C(A^TA)^{-1}C^T)^{-1}(C(A^TA)^{-1}(A^Tb) - d))$$\n"
      ],
      "metadata": {
        "id": "9MRTVbRJOJBL"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 9"
      ],
      "metadata": {
        "id": "ijsygyhnXIQd"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Supporting hyperplane interpretation of KKT conditions**. Consider a convex problem with no equality constraints\n",
        "$$\n",
        "\\begin{split}\n",
        "& f_0(x) \\to \\min\\limits_{x \\in \\mathbb{R}^n }\\\\\n",
        "\\text{s.t. } & f_i(x) \\leq 0, \\quad i = [1,m]\n",
        "\\end{split}\n",
        "$$\n",
        "Assume, that $\\exists x^* \\in \\mathbb{R}^n, \\mu^* \\in \\mathbb{R}^m$ satisfy the KKT conditions\n",
        "$$\n",
        "\\begin{split}\n",
        "& \\nabla_x L (x^*, \\mu^*) = \\nabla f_0(x^*) + \\sum\\limits_{i=1}^m\\mu_i^*\\nabla f_i(x^*) = 0 \\\\\n",
        "& \\mu^*_i \\geq 0, \\quad i = [1,m] \\\\\n",
        "& \\mu^*_i f_i(x^*) = 0, \\quad i = [1,m]\\\\\n",
        "& f_i(x^*) \\leq 0, \\quad i = [1,m]\n",
        "\\end{split}\n",
        "$$\n",
        "Show that\n",
        "$$\n",
        "\\nabla f_0(x^*)^\\top (x - x^*) \\geq 0\n",
        "$$\n",
        "for all feasible $x$. In other words, the KKT conditions imply the simple optimality criterion or $\\nabla f_0(x^*)$ defines a supporting hyperplane to the feasible set at $x^*$."
      ],
      "metadata": {
        "id": "voe84N1wXMng"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "$$ \\nabla f_0(x^*)^\\top (x - x^*) \\xrightarrow{ \\nabla f_0(x^*) + \\sum\\limits_{i=1}^m\\mu_i^*\\nabla f_i(x^*) = 0} = -\\sum\\limits_{i=1}^m\\mu_i^*\\nabla f^T_i(x^*)(x - x^*)$$\n",
        "Для каждого $i$:\n",
        "$$\\mu_i^*\\nabla f^T_i(x - x^*) = \\begin{cases}0 \\leftarrow \\mu_i = 0 | \\text{ Ограничение не активно} \\\\ \\leq 0 \\leftarrow \\nabla f^T_i(x^*)(x - x^*) \\leq 0 | \\text{Ограничение активно} \\end{cases}$$\n",
        "Покажем, что действительно, при активном ограничении, выполняется неравенство. Вспомним, что $f_i$ - выпуклые по условию. Запишем тогда признак первого порядка:\n",
        "$$f_i(x) \\geq f_i(x^*) + \\nabla f^T_i(x^*)(x - x^*) \\to \\nabla f^T_i(x^*)(x - x^*) \\leq f_i(x) - \\overset{=0 \\text{ активное огр.}}{f_i(x^*)} = f_i(x) \\overset{x\\text{ - feasible}}{\\leq} 0$$\n",
        "Тем самым:\n",
        "$$\\mu_i^*\\nabla f^T_i(x - x^*)\\leq 0 \\to -\\sum\\limits_{i=1}^m\\mu_i^*\\nabla f^T_i(x^*)(x - x^*) \\geq 0 \\to \\nabla f_0(x^*)^\\top (x - x^*) \\geq 0$$"
      ],
      "metadata": {
        "id": "R8X-7OobXRju"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 10"
      ],
      "metadata": {
        "id": "QOwepmXPXUAc"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**A penalty method for equality constraints.** We consider the problem of minimization\n",
        "$$\n",
        "\\begin{split}\n",
        "& f_0(x) \\to \\min\\limits_{x \\in \\mathbb{R}^{n} }\\\\\n",
        "\\text{s.t. } & Ax = b,\n",
        "\\end{split}\n",
        "$$\n",
        "where $f_0(x): ^n $ is convex and differentiable, and $A \\in \\mathbb{R}^{m \\times n}$ with $\\mathbf{rank }A = m$. In a quadratic penalty method, we form an auxiliary function\n",
        "$$\n",
        "\\phi(x) = f_0(x) + \\alpha \\|Ax - b\\|_2^2,\n",
        "$$\n",
        "where $\\alpha > 0$ is a parameter. This auxiliary function consists of the objective plus the penalty term $\\alpha \\Vert Ax - b\\Vert_2^2$. The idea is that a minimizer of the auxiliary function, $\\tilde{x}$, should be an approximate solution to the original problem. Intuition suggests that the larger the penalty weight $\\alpha$, the better the approximation $\\tilde{x}$ to a solution of the original problem. Suppose $\\tilde{x}$ is a minimizer of $\\phi(x)$. Show how to find, from $\\tilde{x}$, a dual feasible point for the original problem. Find the corresponding lower bound on the optimal value of the original problem."
      ],
      "metadata": {
        "id": "0Slaa_OJXVRs"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Для того, чтобы понять, как использовать $\\tilde x$ для поиска точки двойственной задачи, попробуем расписать двойственную задачу:\n",
        "\n",
        "$$g(\\nu) = \\inf_{x} (f_0(x) + \\nu^T(Ax-b)) = \\inf_{x} (f_0(x) + \\nu^TAx) - \\nu^T b$$\n",
        "Допустимым будет такой $\\nu$,  что $$g(\\nu) = \\inf_{x} (f_0(x) + \\nu^TAx) - \\nu^T b > -\\infty$$\n",
        "\n",
        "Зная, что $f_0$ - выпуклая и диффернцируемая, утверждаем две вещи. Слейтер выполняется, и я снова повторю, что ограничение равенства в виде афинного преобразования и внутренняя точка есть, если есть решение. Вторая вещь, ККТ РАБОТАЮТ. Выпишем ККТ:\n",
        "$$\\nabla_x L = \\nabla_x f_0(x) + A^T \\nu = 0 \\\\\n",
        "Ax - b = 0$$\n",
        "\n",
        "А теперь внимательно посмотрим на $\\phi$. Заметим, что это строго выпуклая функция, так как квадрат нормы - строго выпуклый, а $f_0$ просто выпуклая. Сумма сохраняет строгость. Тогда в единственном оптимуме будет выполнено:\n",
        "$$\\nabla_x \\phi = \\nabla_x f_0(x) + 2\\alpha A^T(Ax - b) = 0 \\to \\nabla_x \\phi(\\tilde x) = 0 \\to  2\\alpha A^T(A\\tilde x  - b) = -\\nabla_x f_0(\\tilde x) $$\n",
        "Хм, а что же если. Мы возьмем, отсюда $\\tilde \\nu = 2\\alpha(A\\tilde x-b)$. Тогда для $(\\tilde x, \\tilde \\nu)$ выполняется условие стационарности:\n",
        "$$\\nabla_x L = \\nabla_x f_0(\\tilde x) + A^T \\tilde \\nu = 0$$ Из чего:$$\\tilde x =\\text{arg}\\min_x f_0(x) + \\tilde \\nu^T(Ax-b)$$\n",
        "И тогда: $g(\\tilde \\nu) = f_0(\\tilde x) + \\tilde \\nu^T (A \\tilde x - b) > -\\infty$.\n",
        "Кроме того, так как любое допустимое значение двойственной функции - нижняя оценка изначальной задачи. Можем вывести оценку:\n",
        "$$f(x) \\geq g(\\tilde \\nu) = f_0(\\tilde x) + \\tilde \\nu^T (A \\tilde x - b) =  f_0(\\tilde x) + 2\\alpha ||A \\tilde x - b||^2$$"
      ],
      "metadata": {
        "id": "ZnofBW8WXaPq"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Linear programming"
      ],
      "metadata": {
        "id": "AnJShmYXCEHV"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 1"
      ],
      "metadata": {
        "id": "HwxrvWkhC1uh"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "📱🎧💻 Covers manufacturing. Lyzard Corp is producing covers for the following products:\n",
        "\n",
        "    📱 phones\n",
        "    🎧 headphones\n",
        "    💻 laptops\n",
        "\n",
        "The company’s production facilities are such that if we devote the entire production to headphone covers, we can produce 5000 of them in one day. If we devote the entire production to phone covers or laptop covers, we can produce 4000 or 2000 of them in one day.\n",
        "\n",
        "The production schedule is one week (6 working days), and the week’s production must be stored before distribution. Storing 1000 headphone covers (packaging included) takes up 30 cubic feet of space. Storing 1000 phone covers (packaging included) takes up 50 cubic feet of space, and storing 1000 laptop covers (packaging included) takes up 200 cubic feet of space. The total storage space available is 1500 cubic feet.\n",
        "\n",
        "Due to commercial agreements with Lyzard Corp has to deliver at least 6000 headphone covers and 4000 laptop covers per week to strengthen the product’s diffusion.\n",
        "\n",
        "The marketing department estimates that the weekly demand for headphones covers, phone, and laptop covers does not exceed 15000, 12000 and 8000 units, therefore the company does not want to produce more than these amounts for headphones, phone, and laptop covers.\n",
        "\n",
        "Finally, the net profit per headphone cover, phone cover, and laptop cover are $5, $7, and $12, respectively.\n",
        "\n",
        "The aim is to determine a weekly production schedule that maximizes the total net profit."
      ],
      "metadata": {
        "id": "rMlkqsSeC40j"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Write a Linear Programming formulation for the problem. Use the following variables:\n",
        "\n",
        "$y_1$ = number of headphones covers produced over the week,\n",
        "\n",
        "$y_2$ = number of phone covers produced over the week,\n",
        "\n",
        "$y_3$ = number of laptop covers produced over the week."
      ],
      "metadata": {
        "id": "xmFG65SWC6gW"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "$$\\begin{cases} 5y_1 + 7y_2 + 12y_2 \\to \\max_{y_1, y_2, y_3 \\in \\mathbb{R}}\\\\ \\frac{y_1}{5000} + \\frac{y_2}{4000} + \\frac{y_3}{2000} \\leq 6 \\\\\n",
        "30\\frac{y_1}{1000} + 50\\frac{y_2}{1000} + 200\\frac{y_1}{1000} \\leq 1500 \\\\ 6000 \\leq y_1 \\leq 15000 \\\\ 0 \\leq y_2 \\leq 12000 \\\\ 4000 \\leq y_3 \\leq 8000\\end{cases}$$"
      ],
      "metadata": {
        "id": "V4_oxRx-DAiR"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Таргет и ограничения записаны в линейной форме - это подходящая запись задачи линейного программирования."
      ],
      "metadata": {
        "id": "_MzcexFfrz9j"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Find the solution to the problem using PyOMO\n"
      ],
      "metadata": {
        "id": "l0_O-P7LDBBY"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install pyomo\n",
        "! sudo apt-get install glpk-utils --quiet  # GLPK\n",
        "! sudo apt-get install coinor-cbc --quiet  # CoinOR"
      ],
      "metadata": {
        "id": "WIOzlNT0qc1E"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "from pyomo.environ import *\n",
        "\n",
        "model = ConcreteModel()\n",
        "\n",
        "#Для доступа к дуальным переменным\n",
        "model.dual = Suffix(direction=Suffix.IMPORT)\n",
        "\n",
        "#Переменные\n",
        "model.y1 = Var(domain=NonNegativeReals)\n",
        "model.y2 = Var(domain=NonNegativeReals)\n",
        "model.y3 = Var(domain=NonNegativeReals)\n",
        "\n",
        "#Целевая\n",
        "model.profit = Objective(\n",
        "    expr = 5000*model.y1 + 7000*model.y2 + 12000*model.y3,\n",
        "    sense = maximize)\n",
        "\n",
        "#Ограничения\n",
        "model.time = Constraint(expr = model.y1*4 + model.y2*5+ model.y3*10 <= 120)\n",
        "model.space = Constraint(expr = model.y1*30 + model.y2*50+ model.y3*200 <= 1500)\n",
        "model.demand_y1 =   Constraint(expr = model.y1 <= 15)\n",
        "model.demand_y2 =   Constraint(expr = model.y2 <= 12)\n",
        "model.demand_y3 =   Constraint(expr = model.y3 <= 8)\n",
        "model.order_y1 =   Constraint(expr = model.y1 >= 6)\n",
        "model.order_y3 =   Constraint(expr = model.y3 >= 4)\n",
        "\n",
        "SolverFactory('cbc').solve(model).write()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 0
        },
        "id": "mb1fC0Y4qduN",
        "outputId": "fb64198f-b7e3-454b-d69e-0efc6934298e"
      },
      "execution_count": 81,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "# ==========================================================\n",
            "# = Solver Results                                         =\n",
            "# ==========================================================\n",
            "# ----------------------------------------------------------\n",
            "#   Problem Information\n",
            "# ----------------------------------------------------------\n",
            "Problem: \n",
            "- Name: unknown\n",
            "  Lower bound: 154000.0\n",
            "  Upper bound: 154000.0\n",
            "  Number of objectives: 1\n",
            "  Number of constraints: 7\n",
            "  Number of variables: 3\n",
            "  Number of nonzeros: 3\n",
            "  Sense: maximize\n",
            "# ----------------------------------------------------------\n",
            "#   Solver Information\n",
            "# ----------------------------------------------------------\n",
            "Solver: \n",
            "- Status: ok\n",
            "  User time: -1.0\n",
            "  System time: 0.0\n",
            "  Wallclock time: 0.0\n",
            "  Termination condition: optimal\n",
            "  Termination message: Model was solved to optimality (subject to tolerances), and an optimal solution is available.\n",
            "  Statistics: \n",
            "    Branch and bound: \n",
            "      Number of bounded subproblems: None\n",
            "      Number of created subproblems: None\n",
            "    Black box: \n",
            "      Number of iterations: 2\n",
            "  Error rc: 0\n",
            "  Time: 0.023072004318237305\n",
            "# ----------------------------------------------------------\n",
            "#   Solution Information\n",
            "# ----------------------------------------------------------\n",
            "Solution: \n",
            "- number of solutions: 0\n",
            "  number of solutions displayed: 0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Profit = \", model.profit())\n",
        "print(\"Headphone covers per week = \", model.y1()*1000)\n",
        "print(\"Phone covers per week = \", model.y2()*1000)\n",
        "print(\"Laptop covers per week = \", model.y3()*1000)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 0
        },
        "id": "f-cOrh1Bud7O",
        "outputId": "bc727162-32a6-4075-acaa-cd668f6b32c3"
      },
      "execution_count": 82,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Profit =  154000.0\n",
            "Headphone covers per week =  10000.0\n",
            "Phone covers per week =  8000.0\n",
            "Laptop covers per week =  4000.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Perform the sensitivity analysis. Which constraint could be relaxed to increase the profit the most? Prove it numerically."
      ],
      "metadata": {
        "id": "PeoE2TirDOFc"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "str = \"{0:5.2f}    {1:1.2f}    {2:1.2f}    {3:3.2f}\"\n",
        "\n",
        "print(\"value  lslack  uslack    dual\")\n",
        "for c in [model.time, model.space, model.demand_y1, model.demand_y2, model.demand_y3, model.order_y1, model.order_y3]:\n",
        "    print(c, \":\")\n",
        "    print(str.format(c(), c.lslack(), c.uslack(), model.dual[c]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 0
        },
        "id": "uCOe9Cn27YKx",
        "outputId": "e6d2ae33-fe98-4d63-d89c-f5cdee3de9a4"
      },
      "execution_count": 83,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "value  lslack  uslack    dual\n",
            "time :\n",
            "120.00    inf    0.00    800.00\n",
            "space :\n",
            "1500.00    inf    0.00    60.00\n",
            "demand_y1 :\n",
            "10.00    inf    5.00    -0.00\n",
            "demand_y2 :\n",
            " 8.00    inf    4.00    -0.00\n",
            "demand_y3 :\n",
            " 4.00    inf    4.00    -0.00\n",
            "order_y1 :\n",
            "10.00    4.00    inf    -0.00\n",
            "order_y3 :\n",
            " 4.00    0.00    inf    -8000.00\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Итого активные ограничения:\n",
        "1. Время\n",
        "2. Место на складе\n",
        "3. Минимальное количество штук для ноутбуков\n",
        "\n",
        "Самый большой по модулю у последнего ограничения, далее у времени, и наименьший у пространства. (из лекции) При больших коэф-ах при ослаблении ограничения, целевая должна существенно вырасти.\n",
        "\n",
        "Проверим для времени и минимальному количеству ноутбуков:\n"
      ],
      "metadata": {
        "id": "e9j8OGyHKXm2"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Ослабляем на 1000 число ноутбуков:\n",
        "$$y_3 \\geq 3000$$"
      ],
      "metadata": {
        "id": "jxYBq-P2L-eZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model = ConcreteModel()\n",
        "\n",
        "#Для доступа к дуальным переменным\n",
        "model.dual = Suffix(direction=Suffix.IMPORT)\n",
        "\n",
        "#Переменные\n",
        "model.y1 = Var(domain=NonNegativeReals)\n",
        "model.y2 = Var(domain=NonNegativeReals)\n",
        "model.y3 = Var(domain=NonNegativeReals)\n",
        "\n",
        "#Целевая\n",
        "model.profit = Objective(\n",
        "    expr = 5000*model.y1 + 7000*model.y2 + 12000*model.y3,\n",
        "    sense = maximize)\n",
        "\n",
        "#Ограничения\n",
        "model.time = Constraint(expr = model.y1*4 + model.y2*5+ model.y3*10 <= 120)\n",
        "model.space = Constraint(expr = model.y1*30 + model.y2*50+ model.y3*200 <= 1500)\n",
        "model.demand_y1 =   Constraint(expr = model.y1 <= 15)\n",
        "model.demand_y2 =   Constraint(expr = model.y2 <= 12)\n",
        "model.demand_y3 =   Constraint(expr = model.y3 <= 8)\n",
        "model.order_y1 =   Constraint(expr = model.y1 >= 6)\n",
        "model.order_y3 =   Constraint(expr = model.y3 >= 3)\n",
        "\n",
        "SolverFactory('cbc').solve(model).write()\n",
        "print(\"Profit = \", model.profit())\n",
        "print(\"Headphone covers per week = \", model.y1()*1000)\n",
        "print(\"Phone covers per week = \", model.y2()*1000)\n",
        "print(\"Laptop covers per week = \", model.y3()*1000)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 0
        },
        "id": "1gpxiu9JMA1a",
        "outputId": "623bc400-c705-49f9-e5be-1f09bf911a54"
      },
      "execution_count": 84,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "# ==========================================================\n",
            "# = Solver Results                                         =\n",
            "# ==========================================================\n",
            "# ----------------------------------------------------------\n",
            "#   Problem Information\n",
            "# ----------------------------------------------------------\n",
            "Problem: \n",
            "- Name: unknown\n",
            "  Lower bound: 157500.0\n",
            "  Upper bound: 157500.0\n",
            "  Number of objectives: 1\n",
            "  Number of constraints: 7\n",
            "  Number of variables: 3\n",
            "  Number of nonzeros: 3\n",
            "  Sense: maximize\n",
            "# ----------------------------------------------------------\n",
            "#   Solver Information\n",
            "# ----------------------------------------------------------\n",
            "Solver: \n",
            "- Status: ok\n",
            "  User time: -1.0\n",
            "  System time: 0.0\n",
            "  Wallclock time: 0.0\n",
            "  Termination condition: optimal\n",
            "  Termination message: Model was solved to optimality (subject to tolerances), and an optimal solution is available.\n",
            "  Statistics: \n",
            "    Branch and bound: \n",
            "      Number of bounded subproblems: None\n",
            "      Number of created subproblems: None\n",
            "    Black box: \n",
            "      Number of iterations: 1\n",
            "  Error rc: 0\n",
            "  Time: 0.01893758773803711\n",
            "# ----------------------------------------------------------\n",
            "#   Solution Information\n",
            "# ----------------------------------------------------------\n",
            "Solution: \n",
            "- number of solutions: 0\n",
            "  number of solutions displayed: 0\n",
            "Profit =  157500.0\n",
            "Headphone covers per week =  7500.0\n",
            "Phone covers per week =  12000.0\n",
            "Laptop covers per week =  3000.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Доход существенно вырос.\n",
        "\n",
        "Проверим для времени, нарушим трудовой кодекс и заставим пахать 7 дней.\n",
        "$$\\frac{y_1}{5000} + \\frac{y_2}{4000} + \\frac{y_3}{2000} \\leq 7$$"
      ],
      "metadata": {
        "id": "HUfftdCzMPBb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model = ConcreteModel()\n",
        "\n",
        "#Для доступа к дуальным переменным\n",
        "model.dual = Suffix(direction=Suffix.IMPORT)\n",
        "\n",
        "#Переменные\n",
        "model.y1 = Var(domain=NonNegativeReals)\n",
        "model.y2 = Var(domain=NonNegativeReals)\n",
        "model.y3 = Var(domain=NonNegativeReals)\n",
        "\n",
        "#Целевая\n",
        "model.profit = Objective(\n",
        "    expr = 5000*model.y1 + 7000*model.y2 + 12000*model.y3,\n",
        "    sense = maximize)\n",
        "\n",
        "#Ограничения\n",
        "model.time = Constraint(expr = model.y1*4 + model.y2*5+ model.y3*10 <= 140)\n",
        "model.space = Constraint(expr = model.y1*30 + model.y2*50+ model.y3*200 <= 1500)\n",
        "model.demand_y1 =   Constraint(expr = model.y1 <= 15)\n",
        "model.demand_y2 =   Constraint(expr = model.y2 <= 12)\n",
        "model.demand_y3 =   Constraint(expr = model.y3 <= 8)\n",
        "model.order_y1 =   Constraint(expr = model.y1 >= 6)\n",
        "model.order_y3 =   Constraint(expr = model.y3 >= 4)\n",
        "\n",
        "SolverFactory('cbc').solve(model).write()\n",
        "print(\"Profit = \", model.profit())\n",
        "print(\"Headphone covers per week = \", model.y1()*1000)\n",
        "print(\"Phone covers per week = \", model.y2()*1000)\n",
        "print(\"Laptop covers per week = \", model.y3()*1000)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 0
        },
        "id": "Fjjrh2HaMeze",
        "outputId": "0ddeb801-11ed-4337-917d-ab1e7019abff"
      },
      "execution_count": 86,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "# ==========================================================\n",
            "# = Solver Results                                         =\n",
            "# ==========================================================\n",
            "# ----------------------------------------------------------\n",
            "#   Problem Information\n",
            "# ----------------------------------------------------------\n",
            "Problem: \n",
            "- Name: unknown\n",
            "  Lower bound: 158000.0\n",
            "  Upper bound: 158000.0\n",
            "  Number of objectives: 1\n",
            "  Number of constraints: 7\n",
            "  Number of variables: 3\n",
            "  Number of nonzeros: 3\n",
            "  Sense: maximize\n",
            "# ----------------------------------------------------------\n",
            "#   Solver Information\n",
            "# ----------------------------------------------------------\n",
            "Solver: \n",
            "- Status: ok\n",
            "  User time: -1.0\n",
            "  System time: 0.0\n",
            "  Wallclock time: 0.0\n",
            "  Termination condition: optimal\n",
            "  Termination message: Model was solved to optimality (subject to tolerances), and an optimal solution is available.\n",
            "  Statistics: \n",
            "    Branch and bound: \n",
            "      Number of bounded subproblems: None\n",
            "      Number of created subproblems: None\n",
            "    Black box: \n",
            "      Number of iterations: 1\n",
            "  Error rc: 0\n",
            "  Time: 0.017197132110595703\n",
            "# ----------------------------------------------------------\n",
            "#   Solution Information\n",
            "# ----------------------------------------------------------\n",
            "Solution: \n",
            "- number of solutions: 0\n",
            "  number of solutions displayed: 0\n",
            "Profit =  158000.0\n",
            "Headphone covers per week =  15000.0\n",
            "Phone covers per week =  5000.0\n",
            "Laptop covers per week =  4000.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Доход вырос еще сильнее."
      ],
      "metadata": {
        "id": "F7tPmABVMjD2"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 2"
      ],
      "metadata": {
        "id": "ZE-3kBhGDdJc"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Prove the optimality of the solution [10 points]\n",
        "$$\n",
        "x = \\left(\\frac{7}{3} , 0, \\frac{1}{3}\\right)^T\n",
        "$$\n",
        "to the following linear programming problem:\n",
        "$$\n",
        "\\begin{split}\n",
        "& 9x_1 + 3x_2 + 7x_3 \\to \\max\\limits_{x \\in \\mathbb{R}^3 }\\\\\n",
        "\\text{s.t. } & 2x_1 + x_2 + 3x_3 \\leq 6 \\\\\n",
        "& 5x_1 + 4x_2 + x_3 \\leq 12 \\\\\n",
        "& 3x_3 \\leq 1,\\\\\n",
        "& x_1, x_2, x_3 \\geq 0\n",
        "\\end{split}\n",
        "$$\n",
        "but you cannot use any numerical algorithm here."
      ],
      "metadata": {
        "id": "4iZoJw16DeH8"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Воспользуемся признаком оптимальности из симплекс метода:\n",
        "1. Поймем какой тут базис из ограничений, для этого посмотрим какие ограничения активны:\n",
        "$$\\frac{14}{3} + 0 + 1 = \\frac{17}{3} < 6 \\\\\n",
        "\\frac{35}{3} + 0 + \\frac{1}{3} = 12 \\\\\n",
        "3 = 1 \\\\\n",
        "7/3 > 0 \\\\\n",
        "0 = 0 \\\\\n",
        "1/3 > 0$$\n",
        "Базис: $b = \\{2, 3, 5 \\}$\n",
        "2. Отметим, что оно доступный.\n",
        "3. Проверим, условие на скалярные коэффициенты. Сначала приведем задачу в стандартный вид:\n",
        "$$-\\left(\\begin{matrix}\n",
        "9 \\\\\n",
        "3 \\\\\n",
        "7\n",
        "\\end{matrix}\\right)^T \\left(\\begin{matrix}\n",
        "x_1 \\\\\n",
        "x_2 \\\\\n",
        "x_3\n",
        "\\end{matrix}\\right) \\to \\min $$\n",
        "Условие выпишем сразу со взятым базисом:\n",
        "$$A_b x= \\left(\\begin{matrix}\n",
        "5 & 4 & 1 \\\\\n",
        "0 & 0 & 3 \\\\\n",
        "0 & -1 & 0\n",
        "\\end{matrix}\\right) \\left(\\begin{matrix}\n",
        "x_1 \\\\\n",
        "x_2 \\\\\n",
        "x_3\n",
        "\\end{matrix}\\right)  \\leq \\left(\\begin{matrix}\n",
        "12 \\\\\n",
        "1 \\\\\n",
        "0\n",
        "\\end{matrix}\\right)$$\n",
        "Наконец проверим условие:\n",
        "$$\\lambda_b = c^TA^{-1}_b = -\\left(\\begin{matrix}\n",
        "9 \\\\\n",
        "3 \\\\\n",
        "7\n",
        "\\end{matrix}\\right)^T\\left(\\begin{matrix}\n",
        "\\frac{1}{5} & \\frac{-1}{15} & \\frac{4}{5} \\\\\n",
        "0 & 0 & -1 \\\\\n",
        "0 & \\frac{1}{3} & 0\n",
        "\\end{matrix}\\right) = \\left(\\begin{matrix}\n",
        "\\frac{-9}{5} & \\frac{-26}{15} & \\frac{-21}{5}\n",
        "\\end{matrix}\\right): 0 ≽ \\lambda_b$$\n",
        "Выполнено - это оптимум."
      ],
      "metadata": {
        "id": "Z2bSzlFODelN"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Task 3"
      ],
      "metadata": {
        "id": "IysNyFDNSCXr"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Economic interpretation of the dual problem: Suppose a small shop makes wooden toys, where each toy train requires one piece of wood and $2$ tins of paint, while each toy boat requires one piece of wood and $1$ tin of paint. The profit on each toy train is $\\$30$, and the profit on each toy boat is $\\$20$. Given an inventory of $80$ pieces of wood and $100$ tins of paint, how many of each toy should be made to maximize the profit?"
      ],
      "metadata": {
        "id": "b-mjKEgJSFax"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Write out the optimization problem in standard form, writing all constraints as inequalities."
      ],
      "metadata": {
        "id": "aaZRvnHYSF0p"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "$$\\begin{cases}30 y_{\\text{train}} + 20 y_{\\text{boat}} \\to \\max \\\\ y_{\\text{train}} + y_{\\text{boat}} \\leq 80 \\\\\n",
        "2y_{\\text{train}} + y_{\\text{boat}} \\leq 100 \\\\\n",
        "y_{\\text{train}} \\geq 0 \\\\\n",
        "y_{\\text{boat}} \\geq 0\\end{cases} \\to \\begin{cases} \\left( \\begin{matrix} -30\\\\ -20\\end{matrix} \\right)^T \\left( \\begin{matrix} y_{\\text{train}} \\\\ y_{\\text{boat}}\\end{matrix} \\right) \\to \\min \\\\ \\left( \\begin{matrix} 1 & 1\\\\2 & 1\\\\-1 & 0\\\\0& -1\\end{matrix} \\right) \\left( \\begin{matrix} y_{\\text{train}} \\\\ y_{\\text{boat}}\\end{matrix} \\right) \\leq \\left( \\begin{matrix} 80 \\\\100 \\\\ 0 \\\\ 0\\end{matrix} \\right) \\end{cases}$$"
      ],
      "metadata": {
        "id": "vUbrgsRzSL3h"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Sketch the feasible set and determine $p^*$ and $x^*$"
      ],
      "metadata": {
        "id": "oGXrJ1LySL-R"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Будем делать именно для оригинальной задачи, а не стандартной формы."
      ],
      "metadata": {
        "id": "AwGwlQUjfslz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(6, 6))\n",
        "plt.xlabel('Trains')\n",
        "plt.ylabel('Boats')\n",
        "\n",
        "# Wood constraint\n",
        "x = np.array([0, 80])\n",
        "y = 80 - x\n",
        "plt.plot(x, y, 'r', lw=2)\n",
        "plt.fill_between([0, 80, 100], [80, 0,0 ], [0, 0, 0], color='r', alpha=0.15, label='Wood constraint')\n",
        "\n",
        "# Paint constraint\n",
        "x = np.array([0, 50])\n",
        "y = 100 - 2*x\n",
        "plt.plot(x, y, 'b', lw=2)\n",
        "plt.fill_between([0, 50, 100], [100, 0, 0], [0, 0, 0], color='b', alpha=0.15, label='Paint constraint')\n",
        "\n",
        "# Objective level lines\n",
        "x = np.array([0, 80])\n",
        "for p in np.linspace(0, 3600, 10):\n",
        "    y = (p - 30*x)/20\n",
        "    plt.plot(x, y, 'y--')\n",
        "\n",
        "plt.ylim(0, 125)\n",
        "plt.xlim(0, 80)\n",
        "plt.legend()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 560
        },
        "id": "U75yHrqZUga4",
        "outputId": "0c129e3c-1039-4c94-f5a0-8d29b8dbf5fe"
      },
      "execution_count": 95,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.legend.Legend at 0x78312177d8d0>"
            ]
          },
          "metadata": {},
          "execution_count": 95
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 600x600 with 1 Axes>"
            ],
            "image/png": "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Wj0+atQoAAwGAzNnziQwMBBnZ2fOO+88tm7d2mrb69at48wzz8TJyYmQkBAeffTRVkdkGhoamDRpEu7u7oSEhPDqq6+2K/Mvv/zC0KFDcXZ2xt/fv9V3UlVVFZMmTcLHxwdXV1cuvfRScnJyWp5fvHgx3t7e/P7770RHR+Pu7s4ll1xCUVFRq9/zzDPPxM3NDW9vb84991zy8vJYvHgxzzzzDKmpqUiShCRJLUc5JEni3Xff5YorrsDNzY3nn38ei8XClClT6NevHy4uLgwePJg33nij1e9y6623ctVVV7XcHzlyJDNnzuThhx/G19eX4OBgnn766ZbnIyMjAbj66quRJKnlvlIcavRNTU0NkiS1dIGbNm3C29ubM844o+U1Y8aMQaVSsXnz5uNux2AwUFtb2+pHaJun51kkJa0lIOA6+vV7Xuk4DkGl0jBkyCf06TODmJivcXIKVjqSQxC15thGjRrFmjVrWu6vWbOGkSNHMmLEiJbHm5qa2Lx5c0tT8vDDD/PDDz+wZMkSduzYQVRUFBdffDGVlZUAFBYWMnbsWIYOHUpqairvvvsuH330EfPmzWvZz0MPPcS6dev4+eef+eOPP1i7di07duw4YdbffvuNq6++mrFjx5KSksLq1as588wzW56/9dZb2bZtG8uWLWPTpk3IsszYsWNbHblobGzklVde4bPPPmP9+vXk5+cze7ZtQkCz2cxVV13FiBEjSEtLY9OmTUybNg1Jkrj++ut58MEHiY2NpaioiKKiIq6//vqW7T799NNcffXVpKenc/vtt2O1WgkLC+O7775j586dzJ07l8cee4xvv/32hL/jkiVLcHNzY/PmzcyfP59nn32WlStXArQ0fp988glFRUX/aQS7nWwnAPmnn3467vNNTU3yaaedJt90000tjz3//PPyoEGD/vPagIAA+Z133jnutp566ikZ+M9PevqTp/Q79EZWq1U2GsuVjuFwxGfWcb2t1pqamuSdO3fKTU1NrZ9obpblf/6R5fR0Wc7O7vqf9HTb/pqb2539gw8+kN3c3GSTySTX1tbKGo1GLi0tlb/88kt5+PDhsizL8urVq2VAzsvLk+vr62WtVit/8cUXLdswGo1yaGioPH/+fFmWZfmxxx6TBw8eLFut1pbXvP3227K7u7tssVjkuro6WafTyd9++23L8xUVFbKLi4t83333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0KFD2bhxIxBe5/nhhx9y6KGHcuGFF5Kamkrfvn15+eWXa98Px2uRz+djxowZjBo1CiFEWH2eRx11FPPmzWPlypUALFy4kB9//JEzzjgDaLzPMywcXfeH/Px8DMP4h+trq1atWL58uU2qmp7t27cD7PG8d74Xapimybhx4zj66KPp3bs3YJ2ny+UiISFht3VD8TwXL15M//79qaqqIiYmhvfee4+ePXuSk5MTNucIMGfOHP78809+//33f7wXTp/nEUccwfTp0+nWrRvbtm3jnnvu4dhjj2XJkiVhdZ5r167l+eefZ/z48dx22238/vvvXH/99bhcLoYPHx6W16L333+foqIiRowYAYRX3E6cOJGSkhK6d++OqqoYhsEDDzzA0KFDgcb7bmmxSYlD+DBmzBiWLFnCjz/+aLeUJqFbt27k5ORQXFzMf//7X4YPH878+fPtltWobNq0iRtuuIGvvvqKiIgIu+U0KTt/WQJkZGRwxBFH0L59e95++20iIyNtVNa4mKbJoYceypQpUwDo27cvS5Ys4YUXXmD48OE2q2saXn31Vc444wxat25tt5RG5+2332bmzJnMmjWLXr16kZOTw7hx42jdunWjfp4t9vFNcnIyqqr+Yxb0jh07SEtLs0lV07Pz3MLlvMeOHcvHH3/Mt99+y0EHHVQ7npaWhs/no6ioaLf1Q/E8XS4XnTt3pl+/fkydOpXMzEyeeuqpsDrHBQsWkJubyyGHHIKmaWiaxvz585k2bRqaptGqVauwOde/k5CQQNeuXVm9enVYfabp6en07Nlzt7EePXrUPqoKt2vRhg0b+Prrr7niiitqx8Lp85wwYQITJ07k4osvpk+fPgwbNowbb7yRqVOnAo33ebbYpMTlctGvXz/mzZtXO2aaJvPmzaN///42KmtaOnbsSFpa2m7nXVJSwq+//hpS5y2lZOzYsbz33nt88803dOzYcbf3+/Xrh67ru53nihUr2LhxY0id554wTZPq6uqwOseTTz6ZxYsXk5OTU/s69NBDGTp0aO2/w+Vc/05ZWRlr1qwhPT09rD7To48++h9l+itXrqR9+/ZA+FyLdvL666+TmprKWWedVTsWTp9nRUUFirJ7yqCqKqZpAo34eTbKtNwQZc6cOdLtdsvp06fLv/76S44ePVomJCTI7du32y3tgCgtLZXZ2dkyOztbAvLxxx+X2dnZcsOGDVJKq2wrISFBfvDBB3LRokXy3HPPDbkyvGuuuUbGx8fL7777brdyvIqKitp1rr76atmuXTv5zTffyD/++EP2799f9u/f30bV+87EiRPl/Pnz5bp16+SiRYvkxIkTpRBCfvnll1LK8DjHf2PX6hspw+dcb7rpJvndd9/JdevWyf/7v/+TAwYMkMnJyTI3N1dKGT7n+dtvv0lN0+QDDzwgV61aJWfOnCmjoqLkjBkzatcJh2uRlFblZrt27eStt976j/fC5fMcPny4bNOmTW1J8LvvviuTk5PlLbfcUrtOY3yeLTopkVLKp59+WrZr1066XC55+OGHy19++cVuSQfMt99+K4F/vIYPHy6ltEq37rzzTtmqVSvpdrvlySefLFesWGGv6H1kT+cHyNdff712ncrKSnnttdfKxMREGRUVJc8//3y5bds2+0TvB6NGjZLt27eXLpdLpqSkyJNPPrk2IZEyPM7x3/h7UhIu5zp48GCZnp4uXS6XbNOmjRw8ePBu3h3hcp5SSvnRRx/J3r17S7fbLbt37y5feuml3d4Ph2uRlFJ+8cUXEtij9nD5PEtKSuQNN9wg27VrJyMiImSnTp3k7bffLqurq2vXaYzPU0i5ix2bg4ODg4ODg4NNtNg5JQ4ODg4ODg7BhZOUODg4ODg4OAQFTlLi4ODg4ODgEBQ4SYmDg4ODg4NDUOAkJQ4ODg4ODg5BgZOUODg4ODg4OAQFTlLi4ODg4ODgEBQ4SYmDg0PI0KFDB5588km7ZTg4ODQRTlLi4ODQ6Agh9vq6++6792u/v//+O6NHj25csQ4ODkGD4+jq4ODQ6Gzfvr3233PnzuWuu+7arTlbTEwMMTExgNVc0TAMNE1rdp0ODg7BhXOnxMHBodFJS0urfcXHxyOEqF1evnw5sbGxfPbZZ/Tr1w+3282PP/7ImjVrOPfcc2nVqhUxMTEcdthhfP3117vt9++Pb4QQvPLKK5x//vlERUXRpUsXPvzww9r3CwsLGTp0KCkpKURGRtKlSxdef/315vrf4ODgsI84SYmDg4MtTJw4kQcffJBly5aRkZFBWVkZZ555JvPmzSM7O5vTTz+dc845h40bN+51P/fccw8XXXQRixYt4swzz2To0KEUFBQAcOedd/LXX3/x2WefsWzZMp5//nmSk5Ob4/QcHBz2A+d+qYODgy3ce++9nHLKKbXLSUlJZGZm1i7fd999vPfee3z44YeMHTv2X/czYsQILrnkEgCmTJnCtGnT+O233zj99NPZuHEjffv25dBDDwWsOy0ODg7Bi3OnxMHBwRZ2Jgo7KSsr4+abb6ZHjx4kJCQQExPDsmXL6r1TkpGRUfvv6Oho4uLiyM3NBeCaa65hzpw5ZGVlccstt/DTTz81/ok4ODg0Gk5S4uDgYAvR0dG7Ld9888289957TJkyhR9++IGcnBz69OmDz+fb6350Xd9tWQiBaZoAnHHGGWzYsIEbb/z/du7QRqEgCsPoT0jwCBIaQBACAhziGQTqCSqAFjbBIKAEgkYjUChaoA8UPeBwW8DyNhlxTgGTO+7LzWR+8nq9slgsst1um70I0BhRAhTh8Xhks9lktVplPB6n3+/n+Xx+fW6v18t6vc7lcsnpdMr5fP5+WOBfeFMCFGEwGOR2u6Wu67Rarez3+9+Nx18dDofMZrOMRqO83+/c7/cMh8OGJgaaZlMCFOF4PKbb7WY+n6eu6yyXy0yn06/O7HQ62e12mUwmqaoq7XY71+u1oYmBpvk8DQAogk0JAFAEUQIAFEGUAABFECUAQBFECQBQBFECABRBlAAARRAlAEARRAkAUARRAgAUQZQAAEUQJQBAET63KBjraRl0JgAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Пересечение множеств ограничений - фиолетовая область будет доступной областью (неотрицательность учтена отрезанием осей графика). Как мы знаем из симплекс метода оптимум будет в одной из угловых точек. Их тут 4. По линиям уровня мы видим, что оптимально производить в точке пересечений ограничений. Найдем её:\n",
        "$$80 -x = 100-2x \\to x = 20, y = 60$$\n",
        "Профит: $30*20 + 20*60 = 1800$\n",
        "Проверим критерием оптмальности из прошлой таски, что это так:\n",
        "$$b = \\{1, 2\\}\\\\\n",
        "A_b = \\left( \\begin{matrix} 1 & 1\\\\2 & 1\\end{matrix} \\right)  \\\\ 0 ≽ \\lambda_b = \\left(\\begin{matrix}\n",
        "-10 & -10\n",
        "\\end{matrix}\\right) $$\n",
        "Это действительно оптимум. Нанесем на график:"
      ],
      "metadata": {
        "id": "TpVtfrWXWIUH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(6, 6))\n",
        "plt.xlabel('Trains')\n",
        "plt.ylabel('Boats')\n",
        "\n",
        "# Wood constraint\n",
        "x = np.array([0, 80])\n",
        "y = 80 - x\n",
        "plt.plot(x, y, 'r', lw=2)\n",
        "plt.fill_between([0, 80, 100], [80, 0,0 ], [0, 0, 0], color='r', alpha=0.15, label='Wood constraint')\n",
        "\n",
        "# Paint constraint\n",
        "x = np.array([0, 50])\n",
        "y = 100 - 2*x\n",
        "plt.plot(x, y, 'b', lw=2)\n",
        "plt.fill_between([0, 50, 100], [100, 0, 0], [0, 0, 0], color='b', alpha=0.15, label='Paint constraint')\n",
        "\n",
        "# Objective level lines\n",
        "x = np.array([0, 80])\n",
        "for p in np.linspace(0, 3600, 10):\n",
        "    y = (p - 30*x)/20\n",
        "    plt.plot(x, y, 'y--')\n",
        "\n",
        "#Optimum\n",
        "plt.scatter(20, 60, color='green', label='Optimum')\n",
        "\n",
        "plt.ylim(0, 125)\n",
        "plt.xlim(0, 80)\n",
        "plt.legend()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 560
        },
        "id": "3GXUJqEQYIri",
        "outputId": "2bbe7dc9-0985-4815-cbd4-83fa11fbf8de"
      },
      "execution_count": 96,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.legend.Legend at 0x783122051510>"
            ]
          },
          "metadata": {},
          "execution_count": 96
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 600x600 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Find the dual problem, then determine $d^*$ and $\\lambda^*$. Note that we can interpret the Lagrange multipliers $\\lambda_k$ associated with the constraints on wood and paint as the prices for each piece of wood and tin of paint, so that $−d^*$ is how much money would be obtained from selling the inventory for those prices. Strong duality says a buyer should not pay more for the inventory than what the toy store would make by producing and selling toys from it, and that the toy store should not sell the inventory for less than that."
      ],
      "metadata": {
        "id": "UdSocPdbSNfY"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Выпишем Лагранжиан:\n",
        "$$L = -\\left( \\begin{matrix} 30 \\\\ 20\\end{matrix} \\right)^Ty + \\lambda^T\\left[ \\left( \\begin{matrix} 1 & 1\\\\2 & 1\\end{matrix} \\right)y - \\left( \\begin{matrix} 80 \\\\ 100\\end{matrix} \\right)\\right] = $$\n",
        "При $\\lambda^T \\left( \\begin{matrix} 1 & 1\\\\2 & 1\\end{matrix} \\right)-\\left( \\begin{matrix} 30 \\\\ 20\\end{matrix} \\right)^T ≽ 0$ инфимум достигается при $y = 0$, иначе нет нижней границы. Тогда двойственная задача:\n",
        "$$\\begin{cases}g(\\lambda_1, \\lambda_2) = -\\lambda^T\\left( \\begin{matrix} 80 \\\\ 100\\end{matrix} \\right) = -80\\lambda_1 - 100 \\lambda_2 \\to max \\\\ \\lambda^T \\left( \\begin{matrix} 1 & 1\\\\2 & 1\\end{matrix} \\right)-\\left( \\begin{matrix} 30 \\\\ 20\\end{matrix} \\right)^T ≽ 0\\to \\left( \\begin{matrix} \\lambda_1 + 2\\lambda_2 -30 \\\\ \\lambda_1 + \\lambda_2 - 20\\end{matrix} \\right) ≽ 0 \\end{cases}$$\n",
        "\n",
        "Получили аналогичную прошлой задачу. Сразу скажу, что оптимумом является: $\\lambda^* = (10, 10)^T$. Пруф - допустимая, краевая точка (по обоим ограничениям) и:\n",
        "$$0 ≽ \\left(\\begin{matrix}\n",
        "80 & 100\n",
        "\\end{matrix}\\right)\\left(\\begin{matrix}\n",
        "-1 & -2 \\\\\n",
        "-1 & -1\n",
        "\\end{matrix}\\right)^{-1} =\\left(\\begin{matrix}\n",
        "-20 & -60\n",
        "\\end{matrix}\\right)$$\n",
        "И $d^* = -800 -1000=-1800$. И для стандартной формы записи - сильная двойственность выполняется."
      ],
      "metadata": {
        "id": "ck4e63N1SPB_"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "The other interpretation of the Lagrange multipliers is as sensitivities to changes in the constraints. Suppose the toymaker found some more pieces of wood; the $\\lambda_k$ associated with the wood constraint will equal the partial derivative of $−p^*$ with respect to how much more wood became available. Suppose the inventory increases by one piece of wood. Use $\\lambda^*$ to estimate how much the profit would increase, without solving the updated optimization problem. How is this consistent with the price interpretation given above for the Lagrange multipliers? source"
      ],
      "metadata": {
        "id": "DywenboSSPEP"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Тут достаточно тяжело понять условие, но предполагаю, что имеет в виду следующее. Хотим оценить прирост связанный с добавалением единицы дерева. Как мы знаем:\n",
        "$$\\lambda_i = -\\frac{\\delta p^*(0)}{\\delta u_i}$$\n",
        "Где $u_i$ - ослабление ограничения на дерево. Тогда:\n",
        "$$\\lambda_i = -\\frac{\\delta p^*(0)}{\\delta u_i} \\approx -(p^*(1) - p^*(0)) + O(1)$$\n",
        "Тогда при увеличении количества дерева на 1 единицу, ожидается прирост прибыли порядка $\\lambda_1 = 10$.\n",
        "\n",
        "Это же согласуется с интерпертацией теневой цены. Число товара выросло на 1 - следовательно прибыль с продажи увеличилось на $1*\\lambda_1 = 10$."
      ],
      "metadata": {
        "id": "r5gqPELGSQ0q"
      }
    }
  ]
}