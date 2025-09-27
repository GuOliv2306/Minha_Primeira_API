"""
Este módulo contém a construção de endpoints otimizados para o simulador. Baseado na simulação de Monte Carlo, estes
endpoints permitem a execução eficiente de múltiplas simulações com diferentes parâmetros, retornando resultados agregados
de forma estruturada e sanitizada para fácil consumo por clientes externos. Esse código foi construído com base nas escrituras,
Inteligencia Artificial e minhas adaptações pessoais.
"""
from src.sanitize_df import *
import pandas as pd
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Literal

class PassengerPriority(str, Enum):
    WOMEN_CHILDREN_FIRST = "women_children_first"
    CLASS_BASED = "class_based"
    AGE_BASED = "age_based"
    DISABILITY_FIRST = "disability_first"
    RANDOM = "random"
    OPTIMAL_SURVIVAL = "optimal_survival"


class EvacuationConstraints(BaseModel):
    total_lifeboat_capacity: int = Field(1178, description="Capacidade total dos botes")
    max_evacuation_time: int = Field(120, description="Tempo máximo de evacuação em minutos")
    crew_members: int = Field(885, description="Número de tripulantes")
    priority_system: PassengerPriority = Field(PassengerPriority.WOMEN_CHILDREN_FIRST)
    consider_mobility: bool = Field(True, description="Considerar mobilidade reduzida")
    panic_factor: float = Field(0.15, description="Fator de pânico (0-1)")


class EvacuationRequest(BaseModel):
    constraints: EvacuationConstraints
    optimize_for: Literal["max_survivors", "min_time", "fairness", "balanced"] = "max_survivors"
    simulation_runs: int = Field(1000, description="Número de simulações Monte Carlo")


class LifeboatAssignment(BaseModel):
    boat_id: str
    passengers: List[Dict]
    capacity_used: int
    max_capacity: int
    evacuation_order: int
    estimated_launch_time: float


class EvacuationPlan(BaseModel):
    plan_id: str
    total_evacuated: int
    total_time_minutes: float
    survival_rate: float
    fairness_score: float
    lifeboat_assignments: List[LifeboatAssignment]
    evacuation_phases: List[Dict]
    optimization_metrics: Dict
    recommendations: List[str]


@dataclass
class Passenger:
    passenger_id: int
    name: str
    age: float
    sex: str
    pclass: int
    fare: float
    embarked: str
    has_family: bool
    mobility_score: float  # 0-1, onde 1 = totalmente móvel
    priority_score: float


class EvacuationOptimizer:
    def __init__(self):
        # Capacidades reais dos botes do Titanic
        self.lifeboat_capacities = {
            'boat_1': 65, 'boat_2': 65, 'boat_3': 65, 'boat_4': 65,
            'boat_5': 65, 'boat_6': 65, 'boat_7': 65, 'boat_8': 65,
            'boat_9': 65, 'boat_10': 65, 'boat_11': 65, 'boat_12': 65,
            'boat_13': 65, 'boat_14': 65, 'boat_15': 65, 'boat_16': 65,
            'collapsible_a': 47, 'collapsible_b': 47,
            'collapsible_c': 47, 'collapsible_d': 47
        }
    
    def calculate_priority_score(self, passenger: Passenger, priority_system: PassengerPriority) -> float:
        """Calcula pontuação de prioridade baseada no sistema escolhido"""
        score = 0.0
        
        if priority_system == PassengerPriority.WOMEN_CHILDREN_FIRST:
            if passenger.sex == 'female':
                score += 1000
            if passenger.age < 16:
                score += 1500
            if passenger.age > 65:
                score += 200
            # Penalidade leve para classe mais alta (vão por último)
            score -= passenger.pclass * 10
            
        elif priority_system == PassengerPriority.CLASS_BASED:
            score = (4 - passenger.pclass) * 1000
            if passenger.sex == 'female':
                score += 100
            if passenger.age < 16:
                score += 200
                
        elif priority_system == PassengerPriority.AGE_BASED:
            if passenger.age < 16:
                score = 2000
            elif passenger.age > 65:
                score = 1500
            else:
                score = 1000 - passenger.age * 10
                
        elif priority_system == PassengerPriority.OPTIMAL_SURVIVAL:
            # Baseado em análise de dados históricos
            if passenger.sex == 'female':
                score += 800
            if passenger.pclass == 1:
                score += 400
            elif passenger.pclass == 2:
                score += 200
            if passenger.age < 16:
                score += 600
            if passenger.has_family:
                score += 100
                
        # Fator de mobilidade
        score *= passenger.mobility_score
        
        return score
    
    def simulate_evacuation_time(self, passengers: List[Passenger], constraints: EvacuationConstraints) -> float:
        """Simula tempo de evacuação considerando fatores reais"""
        base_time_per_passenger = 0.8  # minutos
        
        # Fatores que afetam o tempo
        panic_multiplier = 1 + constraints.panic_factor
        mobility_factor = np.mean([p.mobility_score for p in passengers])
        crowd_factor = min(2.0, len(passengers) / 500)  # Gargalo por multidão
        
        total_time = (len(passengers) * base_time_per_passenger *
                      panic_multiplier * crowd_factor / mobility_factor)

        return float(min(total_time, constraints.max_evacuation_time))
    
    def assign_lifeboats(self, passengers: List[Passenger], constraints: EvacuationConstraints) -> List[LifeboatAssignment]:
        """Atribui passageiros aos botes otimizando a estratégia escolhida"""
        
        # Ordena passageiros por prioridade
        sorted_passengers = sorted(
            passengers,
            key=lambda p: p.priority_score,
            reverse=True,
        )

        assignments = []
        boat_names = list(self.lifeboat_capacities.keys())
        evacuated_count = 0
        launch_time = 0.0

        for i, boat_name in enumerate(boat_names):
            boat_capacity = self.lifeboat_capacities[boat_name]
            boat_passengers = []
            capacity_used = 0

            # Preenche o bote com passageiros prioritários
            while (capacity_used < boat_capacity and
                   evacuated_count < len(sorted_passengers)):

                passenger = sorted_passengers[evacuated_count]
                boat_passengers.append({
                    "passenger_id": int(passenger.passenger_id),
                    "name": passenger.name,
                    "age": float(passenger.age),
                    "sex": passenger.sex,
                    "pclass": int(passenger.pclass),
                    "priority_score": float(passenger.priority_score),
                    "mobility_score": float(passenger.mobility_score)
                })

                capacity_used += 1
                evacuated_count += 1

            # Calcula tempo de lançamento (botes são preparados em paralelo, mas com delay)
            launch_time += float(np.random.normal(8, 2))  # 8 min ± 2 min para preparar cada bote

            assignment = LifeboatAssignment(
                boat_id=str(boat_name),
                passengers=boat_passengers,
                capacity_used=int(capacity_used),
                max_capacity=int(boat_capacity),
                evacuation_order=int(i + 1),
                estimated_launch_time=float(launch_time)
            )

            assignments.append(assignment)

            # Para quando todos os passageiros foram evacuados
            if evacuated_count >= len(sorted_passengers):
                break

        return assignments
    
    def calculate_fairness_score(self, assignments: List[LifeboatAssignment]) -> float:
        """Calcula pontuação de equidade na evacuação"""
        class_distribution = {1: 0, 2: 0, 3: 0}
        gender_distribution = {'male': 0, 'female': 0}
        total_evacuated = 0
        
        for assignment in assignments:
            for passenger in assignment.passengers:
                class_distribution[passenger['pclass']] += 1
                gender_distribution[passenger['sex']] += 1
                total_evacuated += 1
        
        if total_evacuated == 0:
            return 0.0
        
        # Calcula distribuição proporcional ideal vs real
        class_variance = np.var(list(class_distribution.values()))
        gender_ratio = min(gender_distribution['male'], gender_distribution['female']) / total_evacuated
        
        # Pontuação de 0-1, onde 1 é mais equitativo
        fairness = (1 / (1 + class_variance/100)) * gender_ratio
        return float(min(1.0, fairness))
    
    def optimize_evacuation(self, request: EvacuationRequest, passengers_data: pd.DataFrame) -> EvacuationPlan:
        """Função principal de otimização"""
        
        # Converte dados para objetos Passenger
        passengers = []
        for _, row in passengers_data.iterrows():
            # Simula mobilidade baseada em idade e classe
            age = float(to_python_scalar(row['Age'])) if not pd.isna(row['Age']) else 30.0
            mobility = 1.0 - (max(0, age - 65) / 100)  # Reduz com idade
            if row['Sex'] == 'female' and age < 50:
                mobility = min(1.0, mobility + 0.1)  # Mulheres jovens têm leve vantagem
                
            passenger = Passenger(
                passenger_id=int(to_python_scalar(row['PassengerId'])),
                name=str(row['Name']),
                age=float(age),
                sex=str(row['Sex']),
                pclass=int(to_python_scalar(row['Pclass'])),
                fare=float(to_python_scalar(row['Fare'])) if not pd.isna(row['Fare']) else 0.0,
                embarked=str(row.get('Embarked', 'S') or 'S'),
                has_family=bool(to_python_scalar((row['SibSp'] + row['Parch']) > 0)),
                mobility_score=float(max(0.1, mobility)),
                priority_score=0.0  # Será calculado
            )
            passengers.append(passenger)
        
        # Calcula pontuações de prioridade
        for passenger in passengers:
            passenger.priority_score = self.calculate_priority_score(
                passenger, request.constraints.priority_system
            )
        
        best_plan = None
        best_score = -1
        
        # Executa múltiplas simulações Monte Carlo
        for simulation in range(request.simulation_runs):
            # Adiciona um pouco de aleatoriedade para explorar diferentes soluções
            for passenger in passengers:
                noise = float(np.random.normal(0, 10))  # Ruído pequeno
                passenger.priority_score = float(passenger.priority_score + noise)
            
            # Gera plano de evacuação
            assignments = self.assign_lifeboats(passengers, request.constraints)
            
            # Métricas do plano
            total_evacuated = sum(len(a.passengers) for a in assignments)
            survival_rate = total_evacuated / len(passengers)
            evacuation_time = self.simulate_evacuation_time(passengers, request.constraints)
            fairness_score = self.calculate_fairness_score(assignments)
            
            # Função objetivo baseada na otimização escolhida
            if request.optimize_for == "max_survivors":
                score = survival_rate * 1000 - evacuation_time * 0.1
            elif request.optimize_for == "min_time":
                score = 1000 / (evacuation_time + 1) + survival_rate * 100
            elif request.optimize_for == "fairness":
                score = fairness_score * 1000 + survival_rate * 100
            else:  # balanced
                score = survival_rate * 400 + fairness_score * 300 + (120 - evacuation_time) * 2
            score = float(score)
            
            if score > best_score:
                best_score = score
                best_plan = {
                    'assignments': assignments,
                    'total_evacuated': total_evacuated,
                    'evacuation_time': evacuation_time,
                    'survival_rate': survival_rate,
                    'fairness_score': fairness_score
                }
        
        # Gera recomendações
        recommendations = []
        if best_plan['survival_rate'] < 0.5:
            recommendations.append("Considere aumentar a capacidade dos botes salva-vidas")
        if best_plan['evacuation_time'] > 100:
            recommendations.append("Implemente treinamentos de evacuação mais frequentes")
        if best_plan['fairness_score'] < 0.3:
            recommendations.append("Revise o protocolo de prioridade para maior equidade")
        
        # Fases de evacuação
        phases = [
            {"phase": 1, "description": "Mulheres e crianças primeiro", "duration": "0-30 min"},
            {"phase": 2, "description": "Passageiros com mobilidade reduzida", "duration": "30-60 min"},
            {"phase": 3, "description": "Demais passageiros por ordem de prioridade", "duration": "60+ min"}
        ]
        
        return EvacuationPlan(
            plan_id=f"evacuation_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            total_evacuated=int(best_plan['total_evacuated']),
            total_time_minutes=float(best_plan['evacuation_time']),
            survival_rate=float(best_plan['survival_rate']),
            fairness_score=float(best_plan['fairness_score']),
            lifeboat_assignments=best_plan['assignments'],
            evacuation_phases=phases,
            optimization_metrics={
                "optimize_for": request.optimize_for,
                "simulation_runs": request.simulation_runs,
                "best_score": float(best_score),
                "total_lifeboat_capacity": int(sum(self.lifeboat_capacities.values())),
                "utilization_rate": float(best_plan['total_evacuated'] / sum(self.lifeboat_capacities.values()))
            },
            recommendations=recommendations
        )

# Instância global do otimizador
optimizer = EvacuationOptimizer()

