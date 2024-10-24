

import os

os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

###### Para usar no Colab a biblioteca Firedrake:

#try:
  #  import firedrake
#except ImportError:
  #  !wget "https://fem-on-colab.github.io/releases/firedrake-install-real.sh" -O "/tmp/firedrake-install.sh" && bash "/tmp/firedrake-install.sh"
  # import firedrake
	
num_elements = 2000

mesh = UnitIntervalMesh(num_elements)  # Malha Unidimensional com 2000 elementos.
x, = mesh.coordinates

degree = 1
V = FunctionSpace(mesh, "CG", degree) # Elementos Lagrangeanos de grau 1.
W = V * V # Como são duas variáveis que estamos resolvendo, então precisaremos criar um espaço misto.

# Funções do problema

solution = Function(W) # Armazenará a solução do problema variacional.

TP_novo, TN_novo = split(solution) # TP_novo e TN_novo são as funções que queremos solucionar. Sendo assim, esse split(solution) recupera estas funções no espaço.

p, q = TestFunction(W) # Funções testes.


# Implementação das condições iniciais:

solution_initial = Function(W) # Armazenará as condições iniciais.

TP, TN = solution_initial.split() # TP e TN recebem as condições iniciais, respectivamente.


TP0 = Constant(9.23e6) # N° iniciais de TP (células positivas para antígenos).

xP = Constant(0.6113) # Nível médio de expressão de antígeno positivo homeostático.

dP = Constant(0.06) # Desvio padrão.



TN0 = Constant(0.77e6) # N° iniciais de TN (células negativas para antígenos).

xN = Constant(0.2992) # Nível médio de expressão de antígeno negativo.

dN = Constant(0.0608) # Desvio padrão.


# Condições iniciais:

TP.interpolate((TP0/(dP * sqrt(Constant(2) * pi)) * exp(-Constant(0.5) * ((x - xP)/dP)**Constant(2)))) # TP recebe este valor.

TN.interpolate((TN0/(dN * sqrt(Constant(2) * pi)) * exp(-Constant(0.5)*((x - xN)/dN)**Constant(2)))) # TN recebe este valor.


# Os parâmetros referentes a este paciente:

kP = 0.095 # Coeficiente de transição.

sigma = Constant(kP * (dP)**2) # Coeficiente de difusão.

r = Constant(0.18) # Taxa de crescimento de células tumorais.

gamma = Constant(1.45) # Taxa Citotóxica.

r_min = 0.001 # Taxa de expansão basal das células CAR-T.

p_1 = 3.3 # Taxa de Expansão inicial.

p_2 = 5.91e-4 # Modula a duração da taxa de expansão.

p_3 = 3.1 # Decaimento da taxa de expansão.

A = 2.5e2 # Constante de meia saturação da expansão das células CAR-T.

mu = 0.75 # Taxa de mortalidade de CT.

epsilon = 0.05 # Taxa de conversão de CT para CM.

theta_M = 1e-6 # Coeficiente de conversão de células CAR-T.

alpha = 8.7e-7 # Coeficiente de inativação/inibição de células CAR-T por células tumorais.

mu_M = 0.11 # Taxa de mortalidade de células CAR-T de memória.

a = Constant(1e3) # Constantes de meia saturação da expansão da função de lise.

d = Constant(0.305) # Constante de meia saturação da função f(C_T, T_S).

n = Constant(8) # Coeficiente de Hill usado em g(x).

x_med = Constant(0.42) # Fronteira entre as células T_P e T_N.

CT = 5.6e7 # Condição inicial das células CT.

CM = 0 # Condição inicial das células CM.

kI = Constant(4.67e8) # Constante de meia saturação da pressão da terapia.

gamma_b = Constant(0.1*gamma) # Taxa citotóxica do efeito bystander.


def kt(arg): # Função de Expansão
  return (r_min + (p_1/(1 + (p_2 * arg)**(p_3))))


# Definir as funções que dependem de x:

vx_values =  Function(V)

x_barra = Constant(0)

x_barra.assign(xP - ((xP - xN) * Constant(CT)/(kI + Constant(CT)))) # Função x_barra.

vx_values.interpolate(-kP * (x - Constant(x_barra))) # vx_values recebe esta expressão.

gx_values = Function(V).interpolate((x**n)/((x_med)**n + x**n)) # Definição da função gx_values e recebimento desta expressão. 

taxa_citotoxidade = Function(V) # Definição da função taxa_citotoxidade para recebimento de valores no loop iterativo.

# Parâmetros temporais:

Delta_t = 1e-4 # Passo de tempo.

dias = 60.0001 # Dias analisados.

evolucoes = dias/Delta_t

print('Quantidade de passos de tempo:', evolucoes) # Informa a quantidade de passos no tempo.

Total_time = evolucoes * Delta_t # Dias analisados.

dt = Constant(Delta_t)

# Armazenamento das soluções:

CT_solution = [] # Lista para armazenar a solução de CT a cada passo de tempo.

CT_solution.append(CT) # Armazena o primeiro valor de CT.

CM_solution = [] # Lista para armazenar a solução de CM a cada passo de tempo.

CM_solution.append(CM) # Armazena o primeiro valor de CM.

T_population = [] # Lista para armazenar a solução de T a cada passo de tempo.

integral_T = (assemble(TP*dx) + assemble(TN*dx)) # Cálculo de integração: População total de células tumorais - T.

T_population.append(integral_T) # Armazena o primeiro valor de T.

integral_Ts = (assemble(conditional(gt(x, 0.42), TP, 0)*dx)) # Cálculo de integração: N° de células tumorais que são sensíveis à terapia - TS.

lista_TS = [] # Lista para armazenar a solução de TS a cada passo de tempo.

lista_TS.append(integral_Ts) # Armazena o primeiro valor de TS.

TP_resultado = [] #  Lista para armazenar a solução de TP em tempos específicos.

TN_resultado = [] #  Lista para armazenar a solução de TN em tempos específicos.

TP_initial = (assemble(TP).dat.data) #  Transforma o objeto Firedrake em um vetor.

TN_initial = (assemble(TN).dat.data) #  Transforma o objeto Firedrake em um vetor.

TP_resultado.append(TP_initial) # Armazena o primeiro valor de TP - Condição Inicial.

TN_resultado.append(TN_initial) # Armazena o primeiro valor de TN - Condição Inicial.
			
CART_total = [] # Lista para armazenar os valores do CAR-T total.

total_Cart = CT + CM # Recebe a soma de CT com CM.

CART_total.append(total_Cart) # Armazena o primeiro valor de CT + CM.

lise_lista = [] #  Lista para armazenar os valores da função lise f(CT, TS).

funcao_lise = Constant((Constant(CT)/Constant(integral_Ts))/(d + ((a + Constant(CT))/Constant(integral_Ts)))) # Função Lise.

lise_atual = funcao_lise.dat.data[0] # Recebe o valor numérico da função lise.

lise_lista.append(lise_atual) # Armazena o primeiro valor da função lise.


# Implementação das condições de contorno de Dirichlet:

boundary_value_TP = 0 

boundary_value_TN = 0

TP_bc = DirichletBC(W.sub(0), boundary_value_TP, "on_boundary") # No contorno direito.

TN_bc = DirichletBC(W.sub(1), boundary_value_TN, "on_boundary") # No contorno esquerdo.


# Implementação da Formulação Variacional:

F = (inner(1/dt * (TP_novo - TP), p ) * dx)

F += (-kP * inner(TP_novo, p) * dx)

F += (inner((vx_values * TP_novo.dx(0)), p) * dx) # TP_novo.dx(0) indica grad TP_novo em 1D.

F += (sigma * (inner(grad(TP_novo), grad(p)) * dx))

F += (-r * (inner(TP_novo, p) * dx))

F += (gamma_b * (inner(TP_novo, p) * dx))

F += (-gamma_b * (inner((gx_values * TP_novo), p) * dx))

F += (gamma * funcao_lise * (inner((gx_values * TP_novo), p) * dx))

F += (inner(1/dt * (TN_novo - TN), q) * dx)

F += (-r * ((inner(TN_novo, q)) * dx))

F += (gamma_b * ((inner(TN_novo, q)) * dx))


# Para a inicialização do Solver, temos:

solver_parameters={
  'snes_type': 'newtonls',  # Resolver as equações não lineares usando iterações de Newton-Krylov.
  'snes_rtol': 1e-8, # Tolerância relativa para o solver de Newton.
  'pc_type': 'fieldsplit',    # É usado para equações acopladas o qual divide o problema em sub-problemas.
  'fieldsplit_0': {'ksp_type': 'gmres', # Para o primeiro campo: Para cada iteração de Newton será necessário resolver um sistema linear. Usaremos o GMRES.
					'pc_type': 'ilu', # Pré-condicionador.
					'ksp_rtol': 1e-8}, # Tolerância relativa.
  'fieldsplit_1': {'ksp_type': 'gmres', # Para o segundo campo.
					'pc_type': 'ilu',
					'ksp_rtol': 1e-8}
  }

problem = NonlinearVariationalProblem(F, solution, bcs=[TP_bc, TN_bc]) # Armazenará em solution o resultado de F com as condições de contorno.

solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)

tempo_guardar = [10, 15, 30, 45, 60] # Armazenamento de TP e TN nestes tempos (dias).

epsilon_maquina = np.finfo(float).eps # Menor número positivo detectável pela máquina.

citotox_lista = [] # Lista para armazenar a taxa_citotoxidade.

passo_tempo = 0.1 # Tomaremos este passo de tempo para armazenamento dos resultados.

t = 0 # Inicialização do tempo.

i = 0 # Usado somente no print para mostrar a quant. de passos de tempo.

tol = 1e-8 # Valor da tolerância usado no critério de convergência.

max_iterations = 10 # Usado como limite do critério de convergência.


while t < Total_time: # Iterações no tempo.

	print(f'{i}\t= Passo') # Mostra a quant. de passos de tempo.

	TP_ant = (TP) # A variável TP_ant recebe os dados de TP.
	
	TN_ant = (TN) # A variável TN_ant recebe os dados de TN.

	integral_T_ant = 1e8 # Valor tomado aleatoriamente.

	# Implementação do Euler explícito:
	
	CT_novo = CT + (Delta_t *  (((kt(t)) * (integral_Ts)/(A + integral_Ts) * CT)  - (mu * CT) - (epsilon * CT) + (theta_M * CM * integral_Ts) - (alpha * CT * integral_T)))
	
	CM_novo = CM + (Delta_t * ((epsilon * CT) - (theta_M * CM * integral_Ts) - (mu_M * CM)))
	
	print("CT= ", CT_novo) # Mostra os valores de CT_novo a cada passo de tempo.
	
	print("CM= ", CM_novo) # Mostra os valores de CM_novo a cada passo de tempo.

	k = 0 # Inicialização da variável k.

	while ((abs(integral_T - integral_T_ant))/integral_T) > tol and k < max_iterations: # Critério de Convergência.

		integral_TP_ant = (assemble(TP_ant*dx)) # Cálculo da integral: Recebe esta solução, a partir dos dados de TP_ant (solução anterior).
		
		integral_T_ant = integral_T # Recebe a solução de T.

		if integral_Ts < epsilon_maquina: # Evita divisão por valores muito pequenos.
			funcao_lise.assign(Constant(0))
		else:
			funcao_lise.assign(Constant((Constant(CT)/Constant(integral_Ts))/(d + ((a + Constant(CT))/Constant(integral_Ts))))) # Calcular a função lise.


		solver.solve() # Resolve TP e TN.


		TPsol, TNsol = solution.subfunctions # Atribuição da solução da formulação variacional (solution) a cada variável, TPsol e TNsol.

		TP_ant = TPsol # Atualiza TP_ant com a solução obtida na formulação variacional (TPsol).

		TN_ant = TNsol # Atualiza TP_ant com a solução obtida na formulação variacional (TNsol).

		integral_T = (assemble(TP_ant*dx) + assemble(TN_ant*dx)) # Cálculo da integral de T.
		
		integral_Ts = (assemble(conditional(gt(x, 0.42), TP_ant, 0)*dx)) # Cálculo da integral de TS.

		k += 1

		# fim do loop

	if k >= max_iterations: # Limite máximo de iterações permitidas no critério de convergência.
		print("O limite máximo de iterações foi atingido.")
		break


	taxa_citotoxidade.interpolate((gamma_b * (Constant(1) - gx_values)) + (gamma * gx_values * funcao_lise)) # Recebe o valor da taxa de citotoxicidade.
	
	solution_initial.assign(solution) # Atualiza as condições iniciais com a solução atual.
	
	vetor_TP = assemble(TPsol).dat.data # Recebe o valor numérico de TP atual na forma de vetor.
	
	vetor_TN = assemble(TNsol).dat.data # Recebe o valor numérico de TN atual na forma de vetor.
	
	
	for l in range(len(tempo_guardar)): # Armazena somente os valores em tempos específicos (definidos anteriormente em "tempo_guardar").
		if t == tempo_guardar[l]:
			TP_resultado.append(vetor_TP) 
			TN_resultado.append(vetor_TN)
			
			
	CT = CT_novo # Atualiza CT com o CT atual.
	
	CM = CM_novo # Atualiza CM com o CM atual.
	
	total_Cart = CT + CM # Atualiza o CART-total.
	
	x_barra.assign(xP - ((xP - xN) * Constant(CT)/(kI + Constant(CT)))) # Cálculo do x_barra.
	
	vx_values.interpolate(-kP * (x - Constant(x_barra))) # Atualiza o vx_values a cada passo de tempo.
		
	taxa_citotoxidade_valores = assemble(taxa_citotoxidade).dat.data # Recebe os valores em vetor (modifica o objeto Firedrake).
	
	lise_atual = funcao_lise.dat.data[0] # Recebe o valor numérico da função lise.
		
	if t >= passo_tempo:  # Guardar somente a cada passo 0.1 os seguintes resultados:
		CT_solution.append(CT)
		CM_solution.append(CM)
		T_population.append(integral_T)
		CART_total.append(total_Cart)   # CART_TOTAL
		lise_lista.append(lise_atual)  # Armazenamento dos valores da função lise
		citotox_lista.append(taxa_citotoxidade_valores) # Armazenamento da taxa de citotoxidade
		lista_TS.append(integral_Ts)
		
		
		passo_tempo += 0.1
		
		passo_tempo = round(passo_tempo, 5)

	i += 1
	
	t += Delta_t      
	
	t = round(t, 5)    
	
	# Fim do loop




# Guardar os resultados em doc. txt:


# CT:

output_file = 'CT_G02.txt'

np.savetxt(output_file, CT_solution, fmt='%s', delimiter='\t')
#===========================================================

# CM:

output_file = 'CM_G02.txt'

np.savetxt(output_file, CM_solution, fmt='%s', delimiter='\t')
#===========================================================

# T:

output_file = 'T_G02.txt'

np.savetxt(output_file, T_population, fmt='%s', delimiter='\t')
#===========================================================

#  TP:

data = np.column_stack((TP_resultado))

output_file = 'TP_G02.txt'

np.savetxt(output_file, data, fmt = '%s', delimiter = '\t')
#===========================================================

#  TN:

data = np.column_stack((TN_resultado))

output_file = 'TN_G02.txt'

np.savetxt(output_file, data, fmt = '%s', delimiter = '\t')
#===========================================================

#  CART_total:

output_file = 'CART_total_G02.txt'

np.savetxt(output_file, CART_total, fmt = '%s', delimiter = '\t')
#===========================================================

#  Função lise:

output_file = 'funcao_lise_G02.txt'

np.savetxt(output_file, lise_lista, fmt='%s', delimiter='\t')
#===========================================================

# Taxa de citotoxidade:

data = np.column_stack((citotox_lista))

output_file = 'taxa_citoto_G02.txt'

np.savetxt(output_file, data, fmt = '%s', delimiter = '\t')
#===========================================================
	
#  Integral TS:

output_file = 'TS_G02.txt'

np.savetxt(output_file, lista_TS, fmt = '%s', delimiter = '\t')
