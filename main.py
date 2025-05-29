import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Set, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import heapq
import json
import re
from collections import deque, defaultdict
from enum import Enum
import sympy as sp
from sympy import symbols, Eq, solve, simplify, expand, factor, latex, parse_expr
import random
import copy
import os
import logging
import time
from pathlib import Path
import requests
from urllib.parse import urlparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import statistics
from fractions import Fraction
import ast
import traceback
from tqdm import tqdm
import wandb
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StepType(Enum):
    ALGEBRAIC = "algebraic"
    SUBSTITUTION = "substitution"
    FACTORING = "factoring"
    SOLVING = "solving"
    SIMPLIFICATION = "simplification"
    VERIFICATION = "verification"
    BACKTRACK = "backtrack"
    META_STRATEGY = "meta_strategy"

@dataclass
class MathProblem:
    """Represents a mathematical problem with solution"""
    problem: str
    solution: str
    answer: Union[str, float, int]
    difficulty: str = "unknown"
    category: str = "unknown"
    problem_id: str = ""
    
    def __post_init__(self):
        if not self.problem_id:
            self.problem_id = f"prob_{hash(self.problem) % 1000000}"

@dataclass
class Equation:
    left: str
    right: str
    variables: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        self.variables = self.extract_variables()
    
    def extract_variables(self) -> Set[str]:
        """Extract variables from equation"""
        var_pattern = r'[a-zA-Z][a-zA-Z0-9]*'
        vars_left = set(re.findall(var_pattern, self.left))
        vars_right = set(re.findall(var_pattern, self.right))
        # Filter out mathematical functions
        math_funcs = {'sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'abs', 'ln', 'pi', 'e'}
        return (vars_left | vars_right) - math_funcs
    
    def to_sympy(self):
        """Convert to SymPy equation"""
        try:
            left_expr = parse_expr(self.left)
            right_expr = parse_expr(self.right)
            return Eq(left_expr, right_expr)
        except Exception as e:
            logger.debug(f"Failed to convert to SymPy: {e}")
            return None
    
    def __str__(self):
        return f"{self.left} = {self.right}"

@dataclass 
class MathematicalState:
    """Structured representation of mathematical reasoning state"""
    equations: List[Equation] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    goal: Optional[str] = None
    applied_techniques: Set[StepType] = field(default_factory=set)
    assumptions: List[str] = field(default_factory=list)
    progress_indicators: Dict[str, float] = field(default_factory=dict)
    intermediate_values: Dict[str, Any] = field(default_factory=dict)
    
    def get_feature_vector(self) -> np.ndarray:
        """Convert state to feature vector for RL"""
        features = []
        
        # Basic state features
        features.extend([
            len(self.equations),
            len(self.variables),
            len(self.constraints),
            len(self.applied_techniques),
            len(self.intermediate_values)
        ])
        
        # Technique usage (one-hot encoding)
        technique_vector = [0] * len(StepType)
        for i, technique in enumerate(StepType):
            if technique in self.applied_techniques:
                technique_vector[i] = 1
        features.extend(technique_vector)
        
        # Progress indicators
        features.extend([
            self.progress_indicators.get('goal_proximity', 0.0),
            self.progress_indicators.get('equation_complexity', 0.0),
            self.progress_indicators.get('variable_reduction', 0.0),
            self.progress_indicators.get('solution_confidence', 0.0),
        ])
        
        # Variable solving status
        solved_vars = sum(1 for v in self.variables.values() if self._is_numeric_value(v))
        features.append(solved_vars / max(len(self.variables), 1))
        
        return np.array(features, dtype=np.float32)
    
    def _is_numeric_value(self, value: Any) -> bool:
        """Check if value is numeric"""
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return True
            except ValueError:
                return False
        return False
    
    def validate_mathematical_consistency(self) -> Tuple[bool, str]:
        """Check if current state is mathematically consistent"""
        try:
            for eq in self.equations:
                sympy_eq = eq.to_sympy()
                if sympy_eq is None:
                    return False, f"Invalid equation: {eq}"
                
                # Check if we can evaluate both sides
                try:
                    lhs_val = float(sympy_eq.lhs.evalf())
                    rhs_val = float(sympy_eq.rhs.evalf())
                    if abs(lhs_val - rhs_val) > 1e-10:
                        return False, f"Equation not satisfied: {eq}"
                except:
                    # Equation contains variables, which is fine
                    pass
            
            return True, "State is consistent"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def copy(self):
        """Create a deep copy of the state"""
        return copy.deepcopy(self)

@dataclass
class ReasoningState:
    """Enhanced reasoning state with structured math representation"""
    content: str
    math_state: MathematicalState = field(default_factory=MathematicalState)
    score: float = 0.0
    confidence: float = 0.5
    depth: int = 0
    parent_id: Optional[int] = None
    state_id: int = field(default_factory=lambda: np.random.randint(0, 1000000))
    step_type: Optional[StepType] = None
    is_dead_end: bool = False
    failure_reason: Optional[str] = None
    final_answer: Optional[Union[str, float]] = None
    
    def __lt__(self, other):
        return self.score > other.score

class AnswerExtractor:
    """Extract numerical answers from solution text"""
    
    def __init__(self):
        # Patterns for answer extraction
        self.answer_patterns = [
            r'[Tt]he answer is\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
            r'[Aa]nswer:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
            r'=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)(?:\s|$)',
            r'([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$',
            r'\$([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
            r'([+-]?\d+/\d+)',  # Fractions
            r'([+-]?\d+\s*\d+/\d+)',  # Mixed numbers
        ]
        
        # Word number mappings
        self.word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
            'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
            'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000
        }
    
    def extract_answer(self, solution_text: str) -> Union[float, str, None]:
        """Extract numerical answer from solution text"""
        if not solution_text:
            return None
        
        # Try each pattern
        for pattern in self.answer_patterns:
            matches = re.findall(pattern, solution_text, re.IGNORECASE)
            if matches:
                try:
                    answer_str = matches[-1]  # Take the last match
                    return self._parse_number(answer_str)
                except:
                    continue
        
        # Try word numbers
        words = solution_text.lower().split()
        for word in reversed(words):  # Check from end
            if word in self.word_to_num:
                return float(self.word_to_num[word])
        
        return None
    
    def _parse_number(self, num_str: str) -> Union[float, str]:
        """Parse number string to appropriate type"""
        num_str = num_str.strip()
        
        # Handle fractions
        if '/' in num_str:
            try:
                if ' ' in num_str:  # Mixed number like "2 1/3"
                    parts = num_str.split()
                    whole = int(parts[0])
                    frac = Fraction(parts[1])
                    return float(whole + frac)
                else:
                    return float(Fraction(num_str))
            except:
                pass
        
        # Handle regular numbers
        try:
            if '.' in num_str or 'e' in num_str.lower():
                return float(num_str)
            else:
                return float(int(num_str))
        except:
            return num_str
    
    def normalize_answer(self, answer: Union[str, float, int]) -> Union[float, str]:
        """Normalize answer for comparison"""
        if isinstance(answer, str):
            # Try to convert to number
            extracted = self.extract_answer(answer)
            if extracted is not None:
                return extracted
            return answer.strip().lower()
        
        if isinstance(answer, (int, float)):
            # Round to reasonable precision
            if abs(answer - round(answer)) < 1e-6:
                return float(int(round(answer)))
            return round(float(answer), 6)
        
        return answer

class MathDatasetLoader:
    """Load and manage mathematical reasoning datasets"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.answer_extractor = AnswerExtractor()
    
    def load_gsm8k(self, split: str = "train") -> List[MathProblem]:
        """Load GSM8K dataset"""
        logger.info(f"Loading GSM8K {split} split...")
        
        # Sample GSM8K-style problems (in real implementation, load from dataset files)
        sample_problems = [
            {
                "problem": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much money does she make every day?",
                "solution": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins, so she uses 3 + 4 = 7 eggs. This leaves her with 16 - 7 = 9 eggs to sell. She sells these for $2 each, so she makes 9 × $2 = $18 per day.",
                "answer": 18
            },
            {
                "problem": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts of fiber does it take?",
                "solution": "The robe takes 2 bolts of blue fiber. It takes half that much white fiber, so it takes 2/2 = 1 bolt of white fiber. In total, it takes 2 + 1 = 3 bolts of fiber.",
                "answer": 3
            },
            {
                "problem": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increases the value of the house by 150%. How much profit did he make?",
                "solution": "Josh bought the house for $80,000 and put in $50,000 in repairs, so his total investment was $80,000 + $50,000 = $130,000. The repairs increased the house value by 150%, so the new value is $80,000 + $80,000 × 1.5 = $80,000 + $120,000 = $200,000. His profit is $200,000 - $130,000 = $70,000.",
                "answer": 70000
            },
            {
                "problem": "Solve for x: 2x + 5 = 13",
                "solution": "Starting with 2x + 5 = 13. Subtract 5 from both sides: 2x = 13 - 5 = 8. Divide both sides by 2: x = 8/2 = 4.",
                "answer": 4
            },
            {
                "problem": "If 3y - 7 = 2y + 8, what is the value of y?",
                "solution": "Starting with 3y - 7 = 2y + 8. Subtract 2y from both sides: 3y - 2y - 7 = 8, which gives y - 7 = 8. Add 7 to both sides: y = 8 + 7 = 15.",
                "answer": 15
            },
            {
                "problem": "A rectangle has length 12 and width 8. What is its area?",
                "solution": "The area of a rectangle is length × width. So the area is 12 × 8 = 96.",
                "answer": 96
            },
            {
                "problem": "Tom has 3 times as many apples as Jerry. If Jerry has 7 apples, how many apples do they have together?",
                "solution": "Jerry has 7 apples. Tom has 3 times as many, so Tom has 3 × 7 = 21 apples. Together they have 7 + 21 = 28 apples.",
                "answer": 28
            },
            {
                "problem": "A train travels 60 miles in 2 hours. What is its speed in miles per hour?",
                "solution": "Speed = distance / time. The train travels 60 miles in 2 hours, so its speed is 60 / 2 = 30 miles per hour.",
                "answer": 30
            }
        ]
        
        problems = []
        for i, prob_data in enumerate(sample_problems):
            problem = MathProblem(
                problem=prob_data["problem"],
                solution=prob_data["solution"], 
                answer=prob_data["answer"],
                difficulty="elementary" if i < 4 else "middle_school",
                category="word_problem" if i < 3 else "algebra",
                problem_id=f"gsm8k_{split}_{i}"
            )
            problems.append(problem)
        
        logger.info(f"Loaded {len(problems)} GSM8K problems")
        return problems
    
    def load_math_dataset(self, split: str = "train") -> List[MathProblem]:
        """Load MATH competition dataset"""
        logger.info(f"Loading MATH dataset {split} split...")
        
        # Sample MATH-style problems
        sample_problems = [
            {
                "problem": "Find the value of x such that $\\sqrt{x+7} = 5$.",
                "solution": "We have $\\sqrt{x+7} = 5$. Squaring both sides gives $x + 7 = 25$. Therefore $x = 25 - 7 = 18$.",
                "answer": 18,
                "difficulty": "Level 2",
                "category": "Algebra"
            },
            {
                "problem": "What is the sum of the first 10 positive integers?",
                "solution": "The sum of the first n positive integers is $\\frac{n(n+1)}{2}$. For n=10, this gives $\\frac{10 \\cdot 11}{2} = \\frac{110}{2} = 55$.",
                "answer": 55,
                "difficulty": "Level 1", 
                "category": "Algebra"
            },
            {
                "problem": "If $f(x) = 2x + 3$, what is $f(5)$?",
                "solution": "We substitute $x = 5$ into the function: $f(5) = 2(5) + 3 = 10 + 3 = 13$.",
                "answer": 13,
                "difficulty": "Level 1",
                "category": "Algebra"
            },
            {
                "problem": "Solve the system of equations: $x + y = 10$ and $2x - y = 5$.",
                "solution": "From the first equation, $y = 10 - x$. Substituting into the second equation: $2x - (10 - x) = 5$, which gives $2x - 10 + x = 5$, so $3x = 15$ and $x = 5$. Then $y = 10 - 5 = 5$.",
                "answer": "(5, 5)",
                "difficulty": "Level 2",
                "category": "Algebra"
            }
        ]
        
        problems = []
        for i, prob_data in enumerate(sample_problems):
            problem = MathProblem(
                problem=prob_data["problem"],
                solution=prob_data["solution"],
                answer=prob_data["answer"],
                difficulty=prob_data["difficulty"],
                category=prob_data["category"],
                problem_id=f"math_{split}_{i}"
            )
            problems.append(problem)
        
        logger.info(f"Loaded {len(problems)} MATH problems")
        return problems
    
    def create_custom_dataset(self, problems_data: List[Dict]) -> List[MathProblem]:
        """Create dataset from custom problem data"""
        problems = []
        for i, data in enumerate(problems_data):
            problem = MathProblem(
                problem=data["problem"],
                solution=data.get("solution", ""),
                answer=data["answer"],
                difficulty=data.get("difficulty", "unknown"),
                category=data.get("category", "unknown"),
                problem_id=data.get("problem_id", f"custom_{i}")
            )
            problems.append(problem)
        
        return problems

class AdvancedSymPyEngine:
    """Advanced symbolic mathematics engine with step-by-step solving"""
    
    def __init__(self):
        self.common_variables = set('abcdefghijklmnopqrstuvwxyz')
        self.step_descriptions = []
    
    def step_by_step_solve(self, equation_str: str) -> List[Tuple[str, str]]:
        """Solve equation step by step, returning (description, equation) pairs"""
        steps = []
        
        try:
            # Parse equation
            if '=' not in equation_str:
                return steps
            
            left_str, right_str = equation_str.split('=', 1)
            left_expr = parse_expr(left_str.strip())
            right_expr = parse_expr(right_str.strip())
            equation = Eq(left_expr, right_expr)
            
            steps.append(("Starting equation", str(equation)))
            
            # Find variables
            variables = equation.free_symbols
            if not variables:
                steps.append(("No variables to solve", str(equation)))
                return steps
            
            # Choose primary variable to solve for
            target_var = sorted(variables, key=str)[0]
            
            # Apply solving steps
            current_eq = equation
            
            # Step 1: Expand if needed
            expanded_left = expand(current_eq.lhs)
            expanded_right = expand(current_eq.rhs)
            if expanded_left != current_eq.lhs or expanded_right != current_eq.rhs:
                current_eq = Eq(expanded_left, expanded_right)
                steps.append(("Expanding", str(current_eq)))

            try:
                solutions = solve(current_eq, target_var)
                if solutions:
                    if len(solutions) == 1:
                        steps.append((f"Solving for {target_var}", f"{target_var} = {solutions[0]}"))
                    else:
                        for i, sol in enumerate(solutions):
                            steps.append((f"Solution {i+1}", f"{target_var} = {sol}"))
                else:
                    steps.append(("No solution found", str(current_eq)))
            except Exception as e:
                steps.append(("Could not solve", f"Error: {str(e)}"))
        
        except Exception as e:
            steps.append(("Parsing error", f"Could not parse: {equation_str}"))
        
        return steps
    
    def validate_algebraic_step(self, before: str, after: str, operation: str) -> bool:
        """Verify that an algebraic step is mathematically valid"""
        try:
            # Parse both expressions
            before_expr = parse_expr(before) if '=' not in before else None
            after_expr = parse_expr(after) if '=' not in after else None
            
            if before_expr and after_expr:
                # Check if expressions are equivalent
                diff = simplify(before_expr - after_expr)
                return diff == 0
            
            # For equations, check if they're equivalent
            if '=' in before and '=' in after:
                before_left, before_right = before.split('=')
                after_left, after_right = after.split('=')
                
                before_eq = Eq(parse_expr(before_left), parse_expr(before_right))
                after_eq = Eq(parse_expr(after_left), parse_expr(after_right))
                
                # Check if equations have same solutions
                try:
                    before_vars = before_eq.free_symbols
                    after_vars = after_eq.free_symbols
                    
                    if before_vars == after_vars and before_vars:
                        var = list(before_vars)[0]
                        before_sols = solve(before_eq, var)
                        after_sols = solve(after_eq, var)
                        return set(before_sols) == set(after_sols)
                except:
                    pass
            
            return False
            
        except Exception as e:
            logger.debug(f"Validation error: {e}")
            return False
    
    def generate_valid_algebraic_operations(self, equation: Equation) -> List[Tuple[str, Equation]]:
        """Generate valid algebraic operations that can be applied to equation"""
        operations = []
        
        try:
            sympy_eq = equation.to_sympy()
            if not sympy_eq:
                return operations
            
            # Addition operations
            for val in [1, 2, 5, 10]:
                new_eq = Eq(sympy_eq.lhs + val, sympy_eq.rhs + val)
                operations.append((f"Add {val} to both sides", 
                                Equation(str(new_eq.lhs), str(new_eq.rhs))))
            
            # Subtraction operations  
            for val in [1, 2, 5, 10]:
                new_eq = Eq(sympy_eq.lhs - val, sympy_eq.rhs - val)
                operations.append((f"Subtract {val} from both sides",
                                Equation(str(new_eq.lhs), str(new_eq.rhs))))
            
            # Multiplication operations
            for val in [2, 3, -1]:
                if val != 0:
                    new_eq = Eq(sympy_eq.lhs * val, sympy_eq.rhs * val)
                    operations.append((f"Multiply both sides by {val}",
                                    Equation(str(new_eq.lhs), str(new_eq.rhs))))
            
            # Division operations
            for val in [2, 3, 5]:
                new_eq = Eq(sympy_eq.lhs / val, sympy_eq.rhs / val)
                operations.append((f"Divide both sides by {val}",
                                Equation(str(new_eq.lhs), str(new_eq.rhs))))
            
            # Simplification
            simplified_eq = Eq(simplify(sympy_eq.lhs), simplify(sympy_eq.rhs))
            if simplified_eq != sympy_eq:
                operations.append(("Simplify",
                                Equation(str(simplified_eq.lhs), str(simplified_eq.rhs))))
                
        except Exception as e:
            logger.debug(f"Error generating operations: {e}")
        
        return operations[:5]  # Return top 5 operations

class RLValueNetwork(nn.Module):
    """Neural network for estimating state values"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 2))  # [value, confidence]
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.network(x)
        value = torch.tanh(output[:, 0])  # Value in [-1, 1]
        confidence = torch.sigmoid(output[:, 1])  # Confidence in [0, 1]
        return value, confidence

class RLPolicyNetwork(nn.Module):
    """Neural network for action selection"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        layers = []
        current_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits, dim=-1)

@dataclass
class Experience:
    """Experience for RL training"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    confidence: float = 0.5

class ExperienceBuffer:
    """Buffer for storing and sampling experiences"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def size(self):
        return len(self.buffer)

class RLFeasibilityScorer:
    """RL-enhanced feasibility scorer with multi-faceted evaluation"""
    
    def __init__(self, feature_dim: int, device='cpu'):
        self.device = device
        self.value_network = RLValueNetwork(feature_dim).to(device)
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=0.001)
        self.experience_buffer = ExperienceBuffer()
        self.training_step = 0
        
        # Multi-faceted scoring weights
        self.score_weights = {
            'feasibility': 0.3,
            'novelty': 0.2,
            'progress': 0.3,
            'validity': 0.2
        }
    
    def score_states(self, states: List[ReasoningState]) -> List[Tuple[float, float]]:
        """Score states returning (score, confidence) pairs"""
        if not states:
            return []
        
        features = []
        for state in states:
            # Combine text features with mathematical state features
            text_features = self._extract_text_features(state.content)
            math_features = state.math_state.get_feature_vector()
            combined_features = np.concatenate([text_features, math_features])
            features.append(combined_features)
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            values, confidences = self.value_network(features_tensor)
        
        # Multi-faceted scoring
        final_scores = []
        for i, state in enumerate(states):
            base_score = values[i].item()
            confidence = confidences[i].item()
            
            # Calculate individual score components
            feasibility_score = (base_score + 1) / 2  # Convert from [-1,1] to [0,1]
            novelty_score = self._calculate_novelty_score(state)
            progress_score = self._calculate_progress_score(state)
            validity_score = self._calculate_validity_score(state)
            
            # Weighted combination
            final_score = (
                self.score_weights['feasibility'] * feasibility_score +
                self.score_weights['novelty'] * novelty_score +
                self.score_weights['progress'] * progress_score +
                self.score_weights['validity'] * validity_score
            )
            
            final_scores.append((final_score, confidence))
        
        return final_scores
    
    def _extract_text_features(self, text: str) -> np.ndarray:
        """Extract features from text content"""
        features = []
        
        # Length features
        features.append(len(text))
        features.append(len(text.split()))
        features.append(len(text.split('.')))
        
        # Mathematical content indicators
        math_patterns = [
            r'\d+', r'[a-z]\s*=', r'\+|\-|\*|\/', r'\^', 
            r'sqrt', r'sin|cos|tan', r'log', r'\\frac', r'\$'
        ]
        for pattern in math_patterns:
            features.append(len(re.findall(pattern, text.lower())))
        
        # Reasoning indicators
        reasoning_words = [
            'therefore', 'thus', 'so', 'hence', 'because',
            'substituting', 'solving', 'equation', 'let', 'given',
            'answer', 'solution', 'result'
        ]
        for word in reasoning_words:
            features.append(1 if word in text.lower() else 0)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_novelty_score(self, state: ReasoningState) -> float:
        """Calculate how novel/non-repetitive the reasoning is"""
        # Check for repeated techniques
        technique_diversity = len(state.math_state.applied_techniques) / len(StepType)
        
        # Check for repeated phrases in content
        sentences = state.content.split('.')
        unique_sentences = len(set(s.strip() for s in sentences if s.strip()))
        repetition_penalty = unique_sentences / max(len([s for s in sentences if s.strip()]), 1)
        
        return 0.5 * technique_diversity + 0.5 * repetition_penalty
    
    def _calculate_progress_score(self, state: ReasoningState) -> float:
        """Calculate progress toward solving the problem"""
        progress_indicators = state.math_state.progress_indicators
        
        # Base progress from indicators
        base_progress = progress_indicators.get('goal_proximity', 0.0)
        
        # Variable reduction bonus (fewer unknowns is often progress)
        var_reduction = progress_indicators.get('variable_reduction', 0.0)
        
        # Equation simplification bonus
        eq_simplification = progress_indicators.get('equation_complexity', 0.0)
        
        # Solution confidence
        solution_confidence = progress_indicators.get('solution_confidence', 0.0)
        
        return 0.4 * base_progress + 0.2 * var_reduction + 0.2 * eq_simplification + 0.2 * solution_confidence
    
    def _calculate_validity_score(self, state: ReasoningState) -> float:
        """Calculate mathematical validity of the state"""
        is_valid, _ = state.math_state.validate_mathematical_consistency()
        return 1.0 if is_valid else 0.0
    
    def add_experience(self, state: ReasoningState, reward: float, next_state: Optional[ReasoningState] = None):
        """Add training experience"""
        state_features = np.concatenate([
            self._extract_text_features(state.content),
            state.math_state.get_feature_vector()
        ])
        
        next_state_features = None
        if next_state:
            next_state_features = np.concatenate([
                self._extract_text_features(next_state.content),
                next_state.math_state.get_feature_vector()
            ])
        
        experience = Experience(
            state=state_features,
            action=0,  # Simplified for this implementation
            reward=reward,
            next_state=next_state_features if next_state_features is not None else state_features,
            done=next_state is None,
            confidence=state.confidence
        )
        
        self.experience_buffer.add(experience)
    
    def train_step(self, batch_size: int = 32):
        """Perform one training step"""
        if self.experience_buffer.size() < batch_size:
            return 0.0
        
        experiences = self.experience_buffer.sample(batch_size)
        
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)
        
        # Current value estimates
        current_values, current_confidences = self.value_network(states)
        
        # Target values using temporal difference
        with torch.no_grad():
            next_values, _ = self.value_network(next_states)
            targets = rewards + 0.99 * next_values * (~dones).float()
        
        # Value loss
        value_loss = F.mse_loss(current_values, targets)
        
        # Confidence loss (higher confidence for more certain predictions)
        confidence_targets = torch.abs(current_values - targets)
        confidence_loss = F.mse_loss(current_confidences, confidence_targets)
        
        total_loss = value_loss + 0.1 * confidence_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.training_step += 1
        return total_loss.item()

class MathematicalReasoningGenerator:
    """Advanced reasoning generator with mathematical understanding"""
    
    def __init__(self):
        self.sympy_engine = AdvancedSymPyEngine()
        self.technique_templates = {
            StepType.ALGEBRAIC: [
                "Rearranging the equation: {equation}",
                "Adding {value} to both sides: {equation}",
                "Subtracting {value} from both sides: {equation}",
                "Multiplying both sides by {value}: {equation}",
                "Dividing both sides by {value}: {equation}",
            ],
            StepType.SUBSTITUTION: [
                "Substituting {var} = {value}: {equation}",
                "Let {var} = {expression}, then: {equation}",
                "Using the substitution {var} = {value}: {equation}",
            ],
            StepType.FACTORING: [
                "Factoring the expression: {equation}",
                "Using the common factor {factor}: {equation}",
                "Applying the factoring formula: {equation}",
            ],
            StepType.SOLVING: [
                "Solving for {var}: {equation}",
                "Isolating {var}: {equation}",
                "Finding {var} by solving: {equation}",
            ],
            StepType.SIMPLIFICATION: [
                "Simplifying: {equation}",
                "Combining like terms: {equation}",
                "Reducing the expression: {equation}",
            ],
            StepType.VERIFICATION: [
                "Checking our solution: {verification}",
                "Verifying by substitution: {verification}",
                "Confirming the result: {verification}",
            ]
        }
    
    def generate_continuations(self, state: ReasoningState, n: int) -> List[Tuple[str, MathematicalState, StepType]]:
        """Generate mathematical reasoning continuations"""
        continuations = []
        
        # Analyze current mathematical state
        current_equations = state.math_state.equations
        current_variables = state.math_state.variables
        applied_techniques = state.math_state.applied_techniques
        
        # Generate different types of reasoning steps
        available_techniques = [t for t in StepType if t != StepType.BACKTRACK and t != StepType.META_STRATEGY]
        
        for i in range(n):
            # Choose technique (prefer unexplored ones)
            unused_techniques = [t for t in available_techniques if t not in applied_techniques]
            if unused_techniques and random.random() > 0.3:  # 70% chance to try new technique
                technique = random.choice(unused_techniques)
            else:
                technique = random.choice(available_techniques)
            
            continuation, new_math_state = self._generate_step(state, technique)
            continuations.append((continuation, new_math_state, technique))
        
        return continuations
    
    def _generate_step(self, state: ReasoningState, technique: StepType) -> Tuple[str, MathematicalState]:
        """Generate a specific reasoning step"""
        new_math_state = state.math_state.copy()
        new_math_state.applied_techniques.add(technique)
        
        if not state.math_state.equations:
            # If no equations yet, try to extract from problem statement
            equations = self._extract_equations_from_text(state.content)
            new_math_state.equations.extend(equations)
        
        # Generate specific content based on technique
        if technique == StepType.ALGEBRAIC and new_math_state.equations:
            continuation = self._apply_algebraic_operation(new_math_state.equations[0], new_math_state)
        elif technique == StepType.SOLVING and new_math_state.equations:
            continuation = self._apply_solving_step(new_math_state.equations[0], new_math_state)
        elif technique == StepType.SUBSTITUTION:
            continuation = self._apply_substitution(new_math_state)
        elif technique == StepType.VERIFICATION:
            continuation = self._apply_verification(new_math_state)
        elif technique == StepType.SIMPLIFICATION and new_math_state.equations:
            continuation = self._apply_simplification(new_math_state.equations[0], new_math_state)
        else:
            # Default case
            continuation = f"Applying {technique.value} to the current problem."
        
        # Update progress indicators
        self._update_progress_indicators(new_math_state, state.math_state)
        
        return continuation, new_math_state
    
    def _extract_equations_from_text(self, text: str) -> List[Equation]:
        """Extract equations from problem text"""
        equations = []
        
        # Look for equation patterns
        eq_patterns = [
            r'([^=\n]+)=([^=\n]+)',  # Basic equation pattern
            r'([^=]+)equals([^=\n]+)',  # Word form
        ]
        
        for pattern in eq_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                left, right = match[0].strip(), match[1].strip()
                if left and right and len(left) < 50 and len(right) < 50:  # Reasonable length
                    equations.append(Equation(left, right))
        
        # If no equations found, create from common patterns
        if not equations:
            # Look for simple algebra problems
            solve_patterns = [
                r'(\d*[a-z]\s*[+\-]\s*\d+)\s*=\s*(\d+)',
                r'solve.*?(\d*[a-z]\s*[+\-]\s*\d+)\s*=\s*(\d+)',
            ]
            
            for pattern in solve_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    equations.append(Equation(match.group(1), match.group(2)))
                    break
        
        return equations
    
    def _apply_algebraic_operation(self, eq: Equation, math_state: MathematicalState) -> str:
        """Apply algebraic operation to equation"""
        # Use SymPy engine to generate valid operations
        operations = self.sympy_engine.generate_valid_algebraic_operations(eq)
        
        if operations:
            operation_desc, new_equation = random.choice(operations)
            math_state.equations.append(new_equation)
            return f"{operation_desc}: {new_equation}"
        else:
            # Fallback
            return f"Working with equation: {eq}"
    
    def _apply_solving_step(self, eq: Equation, math_state: MathematicalState) -> str:
        """Apply solving step to equation"""
        target_var = list(eq.variables)[0] if eq.variables else 'x'
        
        try:
            sympy_eq = eq.to_sympy()
            if sympy_eq:
                solutions = solve(sympy_eq, target_var)
                if solutions:
                    solution = solutions[0]
                    math_state.variables[target_var] = float(solution) if solution.is_number else str(solution)
                    math_state.intermediate_values[target_var] = str(solution)
                    return f"Solving for {target_var}: {target_var} = {solution}"
        except Exception as e:
            logger.debug(f"Solving error: {e}")
        
        return f"Attempting to solve for {target_var}"
    
    def _apply_substitution(self, math_state: MathematicalState) -> str:
        """Apply substitution step"""
        if math_state.variables:
            var, value = random.choice(list(math_state.variables.items()))
            return f"Substituting {var} = {value} into the equation"
        
        # Create a substitution if none exists
        var = 'x'
        value = random.randint(1, 10)
        math_state.intermediate_values[var] = value
        return f"Let {var} = {value}"
    
    def _apply_verification(self, math_state: MathematicalState) -> str:
        """Apply verification step"""
        if math_state.variables and math_state.equations:
            var, value = random.choice(list(math_state.variables.items()))
            eq = random.choice(math_state.equations)
            
            try:
                # Verify the solution
                sympy_eq = eq.to_sympy()
                if sympy_eq:
                    substituted = sympy_eq.subs(var, value)
                    if substituted.lhs.equals(substituted.rhs):
                        return f"Verification: Substituting {var} = {value} into {eq} confirms our solution"
                    else:
                        return f"Verification: Substituting {var} = {value} into {eq} shows: {substituted}"
            except:
                pass
            
            return f"Checking: {var} = {value} in equation {eq}"
        
        return "Verifying our solution"
    
    def _apply_simplification(self, eq: Equation, math_state: MathematicalState) -> str:
        """Apply simplification step"""
        try:
            sympy_eq = eq.to_sympy()
            if sympy_eq:
                simplified_left = simplify(sympy_eq.lhs)
                simplified_right = simplify(sympy_eq.rhs)
                
                if simplified_left != sympy_eq.lhs or simplified_right != sympy_eq.rhs:
                    new_eq = Equation(str(simplified_left), str(simplified_right))
                    math_state.equations.append(new_eq)
                    return f"Simplifying: {new_eq}"
        except:
            pass
        
        return f"Simplifying the equation: {eq}"
    
    def _update_progress_indicators(self, new_state: MathematicalState, old_state: MathematicalState):
        """Update progress indicators"""
        # Goal proximity
        solved_vars_new = sum(1 for v in new_state.variables.values() if isinstance(v, (int, float)))
        solved_vars_old = sum(1 for v in old_state.variables.values() if isinstance(v, (int, float)))
        
        if solved_vars_new > solved_vars_old:
            new_state.progress_indicators['goal_proximity'] = 0.8
        elif len(new_state.equations) > len(old_state.equations):
            new_state.progress_indicators['goal_proximity'] = 0.6
        else:
            new_state.progress_indicators['goal_proximity'] = 0.4
        
        # Variable reduction
        if solved_vars_new > solved_vars_old:
            new_state.progress_indicators['variable_reduction'] = 1.0
        else:
            new_state.progress_indicators['variable_reduction'] = 0.5
        
        # Equation complexity
        old_complexity = sum(len(str(eq)) for eq in old_state.equations)
        new_complexity = sum(len(str(eq)) for eq in new_state.equations)
        
        if new_complexity < old_complexity:
            new_state.progress_indicators['equation_complexity'] = 0.8
        else:
            new_state.progress_indicators['equation_complexity'] = 0.3
        
        # Solution confidence
        if solved_vars_new > 0:
            new_state.progress_indicators['solution_confidence'] = 0.9
        elif len(new_state.intermediate_values) > len(old_state.intermediate_values):
            new_state.progress_indicators['solution_confidence'] = 0.6
        else:
            new_state.progress_indicators['solution_confidence'] = 0.3

class BacktrackingManager:
    """Manages intelligent backtracking and failure analysis"""
    
    def __init__(self, max_history: int = 50):
        self.state_history: List[ReasoningState] = []
        self.failure_patterns: Dict[str, int] = defaultdict(int)
        self.max_history = max_history
    
    def add_state(self, state: ReasoningState):
        """Add state to history"""
        self.state_history.append(state)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
    
    def should_backtrack(self, current_states: List[ReasoningState]) -> bool:
        """Determine if backtracking is needed"""
        if not current_states:
            return True
        
        # Check if all current states have low scores
        avg_score = np.mean([s.score for s in current_states])
        if avg_score < 0.2:
            return True
        
        # Check if we're stuck (similar states repeating)
        if len(self.state_history) >= 3:
            recent_contents = [s.content[-100:] for s in self.state_history[-3:]]
            if len(set(recent_contents)) <= 1:  # All very similar
                return True
        
        return False
    
    def find_backtrack_point(self, current_states: List[ReasoningState]) -> Optional[ReasoningState]:
        """Find the best point to backtrack to"""
        if not self.state_history:
            return None
        
        # Find states with high scores and different approaches
        candidates = []
        for state in reversed(self.state_history[-15:]):  # Look at recent history
            if state.score > 0.5 and not state.is_dead_end:
                # Check if this state's approach is different from current failing approach
                current_techniques = set()
                for s in current_states:
                    current_techniques.update(s.math_state.applied_techniques)
                
                state_techniques = state.math_state.applied_techniques
                overlap = len(state_techniques & current_techniques)
                if overlap < len(current_techniques) * 0.6:  # Less than 60% overlap
                    candidates.append((state, state.score))
        
        if candidates:
            # Return the highest scoring candidate
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def analyze_failure(self, failed_state: ReasoningState) -> str:
        """Analyze why a reasoning branch failed"""
        failure_reasons = []
        
        # Check mathematical consistency
        is_valid, validity_msg = failed_state.math_state.validate_mathematical_consistency()
        if not is_valid:
            failure_reasons.append(f"Mathematical inconsistency: {validity_msg}")
        
        # Check for repetitive patterns
        if len(failed_state.math_state.applied_techniques) < 2:
            failure_reasons.append("Insufficient technique diversity")
        
        # Check progress
        progress = failed_state.math_state.progress_indicators.get('goal_proximity', 0)
        if progress < 0.3:
            failure_reasons.append("Low progress toward goal")
        
        # Check if stuck in loop
        if failed_state.depth > 8 and progress < 0.4:
            failure_reasons.append("Reasoning loop detected")
        
        # Record pattern for future avoidance
        pattern = f"{failed_state.step_type}_{len(failed_state.math_state.applied_techniques)}"
        self.failure_patterns[pattern] += 1
        
        if not failure_reasons:
            failure_reasons.append("Low feasibility score")
        
        return "; ".join(failure_reasons)
    
    def suggest_alternative_strategy(self, failed_states: List[ReasoningState]) -> List[StepType]:
        """Suggest alternative strategies based on failure analysis"""
        # Analyze what techniques have been failing
        failed_techniques = set()
        for state in failed_states:
            failed_techniques.update(state.math_state.applied_techniques)
        
        # Suggest techniques that haven't been tried or have low failure rates
        all_techniques = list(StepType)
        suggestions = []
        
        for technique in all_techniques:
            if technique not in failed_techniques:
                suggestions.append(technique)
        
        # If all techniques have been tried, suggest ones with lowest failure rates
        if not suggestions:
            technique_failure_rates = {}
            for pattern, count in self.failure_patterns.items():
                if '_' in pattern:
                    technique_str = pattern.split('_')[0]
                    try:
                        technique = StepType(technique_str)
                        technique_failure_rates[technique] = technique_failure_rates.get(technique, 0) + count
                    except:
                        pass
            
            suggestions = sorted(all_techniques, key=lambda t: technique_failure_rates.get(t, 0))[:3]
        
        return suggestions[:3]  # Return top 3 suggestions

class ChainOfThoughtBaseline:
    """Standard Chain of Thought baseline for comparison"""
    
    def __init__(self):
        self.answer_extractor = AnswerExtractor()
    
    def solve(self, problem: str, max_steps: int = 5) -> Dict:
        """Solve using standard chain of thought"""
        reasoning_steps = []
        current_text = problem
        
        # Generate reasoning steps
        for step in range(max_steps):
            next_step = self._generate_next_step(current_text, step)
            if not next_step:
                break
                
            reasoning_steps.append(next_step)
            current_text += " " + next_step
        
        # Extract final answer
        final_answer = self.answer_extractor.extract_answer(current_text)
        
        return {
            "reasoning_chain": " ".join(reasoning_steps),
            "final_answer": final_answer,
            "steps_taken": len(reasoning_steps),
            "method": "chain_of_thought"
        }
    
    def _generate_next_step(self, current_text: str, step_num: int) -> str:
        """Generate next reasoning step (simplified)"""
        # This is a simplified version - in practice would use a language model
        if "solve for" in current_text.lower() and "=" in current_text:
            if step_num == 0:
                return "Let me solve this equation step by step."
            elif step_num == 1:
                return "First, I'll isolate the variable by moving terms to one side."
            elif step_num == 2:
                return "Then I'll simplify the expression."
            elif step_num == 3:
                return "Finally, I'll calculate the final value."
        
        if step_num == 0:
            return "Let me work through this problem systematically."
        elif step_num == 1:
            return "I'll identify the key information and what needs to be found."
        elif step_num == 2:
            return "Now I'll apply the appropriate mathematical operations."
        
        return ""

class TreeOfThoughtsBaseline:
    """Tree of Thoughts baseline for comparison"""
    
    def __init__(self):
        self.answer_extractor = AnswerExtractor()
    
    def solve(self, problem: str, breadth: int = 3, depth: int = 4) -> Dict:
        """Solve using tree of thoughts approach"""
        # Initialize with problem
        current_level = [problem]
        all_paths = []
        
        for level in range(depth):
            next_level = []
            
            for thought in current_level:
                # Generate candidate next thoughts
                candidates = self._generate_candidates(thought, breadth)
                next_level.extend(candidates)
            
            # Keep best candidates
            scored_thoughts = [(t, self._score_thought(t)) for t in next_level]
            scored_thoughts.sort(key=lambda x: x[1], reverse=True)
            current_level = [t[0] for t in scored_thoughts[:breadth]]
            
            all_paths.extend(current_level)
        
        # Find best final answer
        best_path = max(current_level, key=self._score_thought)
        final_answer = self.answer_extractor.extract_answer(best_path)
        
        return {
            "reasoning_chain": best_path,
            "final_answer": final_answer,
            "paths_explored": len(all_paths),
            "method": "tree_of_thoughts"
        }
    
    def _generate_candidates(self, current_thought: str, n: int) -> List[str]:
        """Generate candidate next thoughts"""
        candidates = []
        base_continuations = [
            "Let me approach this differently by",
            "Another way to think about this is",
            "I can also solve this by",
            "Alternatively, I could",
            "Let me try a different method:"
        ]
        
        for i in range(min(n, len(base_continuations))):
            continuation = base_continuations[i] + " working through the algebra step by step."
            candidates.append(current_thought + " " + continuation)
        
        return candidates
    
    def _score_thought(self, thought: str) -> float:
        """Score a thought for quality (simplified)"""
        score = 0.5  # Base score
        
        # Length penalty for very short or very long thoughts
        words = len(thought.split())
        if 20 <= words <= 100:
            score += 0.2
        
        # Bonus for mathematical content
        if re.search(r'\d+|\+|\-|\*|\/|=', thought):
            score += 0.3
        
        # Bonus for reasoning words
        reasoning_words = ['therefore', 'because', 'so', 'thus', 'hence']
        for word in reasoning_words:
            if word in thought.lower():
                score += 0.1
                break
        
        return score

class MathMetrics:
    """Comprehensive evaluation metrics for mathematical reasoning"""
    
    def __init__(self):
        self.answer_extractor = AnswerExtractor()
    
    def calculate_accuracy(self, predictions: List[Union[str, float]], 
                         ground_truth: List[Union[str, float]]) -> float:
        """Calculate accuracy of predictions"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        correct = 0
        for pred, true in zip(predictions, ground_truth):
            if self._answers_match(pred, true):
                correct += 1
        
        return correct / len(predictions)
    
    def _answers_match(self, pred: Union[str, float], true: Union[str, float]) -> bool:
        """Check if predicted answer matches ground truth"""
        # Normalize both answers
        pred_norm = self.answer_extractor.normalize_answer(pred)
        true_norm = self.answer_extractor.normalize_answer(true)
        
        # Handle numeric comparison
        if isinstance(pred_norm, (int, float)) and isinstance(true_norm, (int, float)):
            return abs(pred_norm - true_norm) < 1e-6
        
        # Handle string comparison
        if isinstance(pred_norm, str) and isinstance(true_norm, str):
            return pred_norm.lower().strip() == true_norm.lower().strip()
        
        # Handle mixed types
        try:
            pred_float = float(pred_norm) if not isinstance(pred_norm, (int, float)) else pred_norm
            true_float = float(true_norm) if not isinstance(true_norm, (int, float)) else true_norm
            return abs(pred_float - true_float) < 1e-6
        except:
            return str(pred_norm).lower() == str(true_norm).lower()
    
    def calculate_reasoning_quality(self, solution_paths: List[str]) -> Dict[str, float]:
        """Measure reasoning step quality"""
        metrics = {}
        
        # Average length
        lengths = [len(path.split()) for path in solution_paths]
        metrics['avg_reasoning_length'] = np.mean(lengths)
        metrics['std_reasoning_length'] = np.std(lengths)
        
        # Mathematical content density
        math_densities = []
        for path in solution_paths:
            math_matches = len(re.findall(r'\d+|\+|\-|\*|\/|=', path))
            total_words = len(path.split())
            density = math_matches / max(total_words, 1)
            math_densities.append(density)
        
        metrics['avg_math_density'] = np.mean(math_densities)
        
        # Reasoning word usage
        reasoning_words = ['therefore', 'because', 'so', 'thus', 'hence', 'since']
        reasoning_usage = []
        for path in solution_paths:
            count = sum(1 for word in reasoning_words if word in path.lower())
            reasoning_usage.append(count)
        
        metrics['avg_reasoning_words'] = np.mean(reasoning_usage)
        
        return metrics
    
    def calculate_efficiency(self, solution_paths: List[str], 
                           correct_predictions: List[bool]) -> Dict[str, float]:
        """Measure reasoning efficiency"""
        metrics = {}
        
        # Steps per correct solution
        correct_steps = [len(path.split('.')) for path, correct in 
                        zip(solution_paths, correct_predictions) if correct]
        
        if correct_steps:
            metrics['avg_steps_correct'] = np.mean(correct_steps)
            metrics['min_steps_correct'] = min(correct_steps)
            metrics['max_steps_correct'] = max(correct_steps)
        else:
            metrics['avg_steps_correct'] = 0
            metrics['min_steps_correct'] = 0
            metrics['max_steps_correct'] = 0
        
        # Success rate
        metrics['success_rate'] = sum(correct_predictions) / len(correct_predictions)
        
        return metrics

@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Model parameters
    model_params: Dict = field(default_factory=lambda: {
        'feature_dim': 50,
        'n_candidates': 4,
        'k_survivors': 2,
        'max_depth': 10,
        'min_score_threshold': 0.2
    })
    
    # Training parameters
    training_params: Dict = field(default_factory=lambda: {
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'training_frequency': 10
    })
    
    # Evaluation parameters
    evaluation_params: Dict = field(default_factory=lambda: {
        'datasets': ['gsm8k'],
        'max_problems': 20,
        'compare_baselines': True
    })
    
    # Experiment metadata
    experiment_name: str = "artemis_experiment"
    output_dir: str = "./results"
    use_wandb: bool = False

@dataclass
class ExperimentResults:
    """Results from an experiment"""
    accuracy: float
    reasoning_quality: Dict[str, float]
    efficiency_metrics: Dict[str, float]
    baseline_comparisons: Dict[str, Dict[str, float]]
    training_metrics: Dict[str, List[float]]
    config: ExperimentConfig
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class ExperimentRunner:
    """Run and manage experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_loader = MathDatasetLoader()
        self.metrics = MathMetrics()
        self.answer_extractor = AnswerExtractor()
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(project="artemis-math-reasoning", config=config.__dict__)
    
    def run_experiment(self) -> ExperimentResults:
        """Run complete experiment"""
        logger.info(f"Starting experiment: {self.config.experiment_name}")
        
        # Load datasets
        datasets = {}
        for dataset_name in self.config.evaluation_params['datasets']:
            if dataset_name == 'gsm8k':
                datasets[dataset_name] = self.data_loader.load_gsm8k('train')
            elif dataset_name == 'math':
                datasets[dataset_name] = self.data_loader.load_math_dataset('train')
        
        # Initialize model
        artemis = RLEnhancedAdaptiveCoT(**self.config.model_params)
        
        # Training phase
        logger.info("Starting training phase...")
        training_metrics = self._train_model(artemis, datasets)
        
        # Evaluation phase
        logger.info("Starting evaluation phase...")
        eval_results = self._evaluate_model(artemis, datasets)
        
        # Baseline comparison
        baseline_results = {}
        if self.config.evaluation_params['compare_baselines']:
            logger.info("Running baseline comparisons...")
            baseline_results = self._compare_baselines(datasets)
        
        # Compile results
        results = ExperimentResults(
            accuracy=eval_results['accuracy'],
            reasoning_quality=eval_results['reasoning_quality'],
            efficiency_metrics=eval_results['efficiency_metrics'],
            baseline_comparisons=baseline_results,
            training_metrics=training_metrics,
            config=self.config
        )
        
        # Save results
        self._save_results(results)
        
        logger.info(f"Experiment completed. Results saved to {self.output_dir}")
        return results
    
    def _train_model(self, model: 'RLEnhancedAdaptiveCoT', 
                    datasets: Dict[str, List[MathProblem]]) -> Dict[str, List[float]]:
        """Train the model"""
        training_metrics = {
            'loss': [],
            'accuracy': [],
            'avg_score': []
        }
        
        all_problems = []
        for problems in datasets.values():
            all_problems.extend(problems)
        
        num_epochs = self.config.training_params['num_epochs']
        max_problems = self.config.evaluation_params['max_problems']
        
        for epoch in range(num_epochs):
            logger.info(f"Training epoch {epoch + 1}/{num_epochs}")
            
            # Sample subset of problems for training
            epoch_problems = random.sample(all_problems, min(max_problems, len(all_problems)))
            epoch_loss = []
            epoch_correct = 0
            epoch_scores = []
            
            for problem in tqdm(epoch_problems, desc=f"Epoch {epoch + 1}"):
                try:
                    # Solve problem
                    result = model.solve(problem.problem)
                    
                    # Calculate reward
                    predicted_answer = result['best_solution'].get('final_answer') or \
                                     self.answer_extractor.extract_answer(result['best_solution']['content'])
                    
                    reward = 1.0 if self._answers_match(predicted_answer, problem.answer) else -0.5
                    
                    # Add experience and train
                    if hasattr(model, 'scorer'):
                        # Create dummy state for training
                        dummy_state = ReasoningState(content=problem.problem)
                        model.scorer.add_experience(dummy_state, reward)
                        
                        if len(model.scorer.experience_buffer.buffer) > 32:
                            loss = model.scorer.train_step()
                            if loss > 0:
                                epoch_loss.append(loss)
                    
                    # Track metrics
                    if self._answers_match(predicted_answer, problem.answer):
                        epoch_correct += 1
                    
                    epoch_scores.append(result['best_solution']['score'])
                    
                except Exception as e:
                    logger.warning(f"Error training on problem: {e}")
                    continue
            
            # Record epoch metrics
            if epoch_loss:
                training_metrics['loss'].append(np.mean(epoch_loss))
            training_metrics['accuracy'].append(epoch_correct / len(epoch_problems))
            training_metrics['avg_score'].append(np.mean(epoch_scores))
            
            logger.info(f"Epoch {epoch + 1} - Acc: {epoch_correct/len(epoch_problems):.3f}, "
                       f"Avg Score: {np.mean(epoch_scores):.3f}")
            
            # Log to wandb if enabled
            if self.config.use_wandb:
                wandb.log({
                    'train_accuracy': epoch_correct / len(epoch_problems),
                    'train_avg_score': np.mean(epoch_scores),
                    'train_loss': np.mean(epoch_loss) if epoch_loss else 0,
                    'epoch': epoch
                })
        
        return training_metrics
    
    def _evaluate_model(self, model: 'RLEnhancedAdaptiveCoT', 
                       datasets: Dict[str, List[MathProblem]]) -> Dict:
        """Evaluate the trained model"""
        all_problems = []
        for problems in datasets.values():
            all_problems.extend(problems)
        
        max_problems = self.config.evaluation_params['max_problems']
        eval_problems = random.sample(all_problems, min(max_problems, len(all_problems)))
        
        predictions = []
        ground_truth = []
        solution_paths = []
        correct_predictions = []
        
        logger.info(f"Evaluating on {len(eval_problems)} problems...")
        
        for problem in tqdm(eval_problems, desc="Evaluating"):
            try:
                result = model.solve(problem.problem)
                
                # Extract answer
                predicted_answer = result['best_solution'].get('final_answer') or \
                                 self.answer_extractor.extract_answer(result['best_solution']['content'])
                
                predictions.append(predicted_answer)
                ground_truth.append(problem.answer)
                solution_paths.append(result['best_solution']['content'])
                
                is_correct = self._answers_match(predicted_answer, problem.answer)
                correct_predictions.append(is_correct)
                
            except Exception as e:
                logger.warning(f"Error evaluating problem: {e}")
                predictions.append(None)
                ground_truth.append(problem.answer)
                solution_paths.append("")
                correct_predictions.append(False)
        
        # Calculate metrics
        accuracy = self.metrics.calculate_accuracy(predictions, ground_truth)
        reasoning_quality = self.metrics.calculate_reasoning_quality(solution_paths)
        efficiency_metrics = self.metrics.calculate_efficiency(solution_paths, correct_predictions)
        
        return {
            'accuracy': accuracy,
            'reasoning_quality': reasoning_quality,
            'efficiency_metrics': efficiency_metrics
        }
    
    def _compare_baselines(self, datasets: Dict[str, List[MathProblem]]) -> Dict[str, Dict[str, float]]:
        """Compare with baseline methods"""
        all_problems = []
        for problems in datasets.values():
            all_problems.extend(problems)
        
        max_problems = min(10, len(all_problems))  # Smaller subset for baselines
        eval_problems = random.sample(all_problems, max_problems)
        
        baselines = {
            'chain_of_thought': ChainOfThoughtBaseline(),
            'tree_of_thoughts': TreeOfThoughtsBaseline()
        }
        
        baseline_results = {}
        
        for baseline_name, baseline_model in baselines.items():
            logger.info(f"Evaluating baseline: {baseline_name}")
            
            predictions = []
            ground_truth = []
            solution_paths = []
            
            for problem in tqdm(eval_problems, desc=f"Baseline {baseline_name}"):
                try:
                    result = baseline_model.solve(problem.problem)
                    predictions.append(result['final_answer'])
                    ground_truth.append(problem.answer)
                    solution_paths.append(result['reasoning_chain'])
                except Exception as e:
                    logger.warning(f"Baseline error: {e}")
                    predictions.append(None)
                    ground_truth.append(problem.answer)
                    solution_paths.append("")
            
            # Calculate metrics
            accuracy = self.metrics.calculate_accuracy(predictions, ground_truth)
            correct_predictions = [self._answers_match(p, t) for p, t in zip(predictions, ground_truth)]
            reasoning_quality = self.metrics.calculate_reasoning_quality(solution_paths)
            efficiency_metrics = self.metrics.calculate_efficiency(solution_paths, correct_predictions)
            
            baseline_results[baseline_name] = {
                'accuracy': accuracy,
                **reasoning_quality,
                **efficiency_metrics
            }
        
        return baseline_results
    
    def _answers_match(self, pred: Union[str, float], true: Union[str, float]) -> bool:
        """Check if answers match"""
        return self.metrics._answers_match(pred, true)
    
    def _save_results(self, results: ExperimentResults):
        """Save experiment results"""
        # Save as JSON
        results_dict = {
            'accuracy': results.accuracy,
            'reasoning_quality': results.reasoning_quality,
            'efficiency_metrics': results.efficiency_metrics,
            'baseline_comparisons': results.baseline_comparisons,
            'training_metrics': results.training_metrics,
            'config': results.config.__dict__,
            'timestamp': results.timestamp
        }
        
        output_file = self.output_dir / f"{self.config.experiment_name}_results.json"
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Create summary report
        self._create_summary_report(results)
    
    def _create_summary_report(self, results: ExperimentResults):
        """Create human-readable summary report"""
        report = f"""
# ARTEMIS Experiment Results

**Experiment:** {self.config.experiment_name}
**Timestamp:** {results.timestamp}

## Main Results

**Accuracy:** {results.accuracy:.3f}

## Reasoning Quality
- Average reasoning length: {results.reasoning_quality.get('avg_reasoning_length', 0):.1f} words
- Math content density: {results.reasoning_quality.get('avg_math_density', 0):.3f}
- Reasoning words per solution: {results.reasoning_quality.get('avg_reasoning_words', 0):.1f}

## Efficiency Metrics
- Success rate: {results.efficiency_metrics.get('success_rate', 0):.3f}
- Average steps (correct): {results.efficiency_metrics.get('avg_steps_correct', 0):.1f}

## Baseline Comparisons
"""
        
        for baseline_name, metrics in results.baseline_comparisons.items():
            report += f"\n### {baseline_name.title()}\n"
            report += f"- Accuracy: {metrics.get('accuracy', 0):.3f}\n"
            report += f"- Success rate: {metrics.get('success_rate', 0):.3f}\n"
        
        report += f"\n## Training Progress\n"
        if results.training_metrics.get('accuracy'):
            final_train_acc = results.training_metrics['accuracy'][-1]
            report += f"- Final training accuracy: {final_train_acc:.3f}\n"
        
        # Save report
        report_file = self.output_dir / f"{self.config.experiment_name}_report.md"
        with open(report_file, 'w') as f:
            f.write(report)

class RLEnhancedAdaptiveCoT:
    """RL-Enhanced Adaptive Hypothesis-Pruning Chain of Thought with full improvements"""
    
    def __init__(self,
                 feature_dim: int = 50,
                 n_candidates: int = 5,
                 k_survivors: int = 2,
                 max_depth: int = 15,
                 min_score_threshold: float = 0.15,
                 device='cpu'):
        
        # Initialize components
        self.generator = MathematicalReasoningGenerator()
        self.scorer = RLFeasibilityScorer(feature_dim, device)
        self.backtrack_manager = BacktrackingManager()
        self.answer_extractor = AnswerExtractor()
        
        # Parameters
        self.n_candidates = n_candidates
        self.k_survivors = k_survivors
        self.max_depth = max_depth
        self.min_score_threshold = min_score_threshold
        
        # State management
        self.active_states = []
        self.completed_states = []
        self.pruning_history = []
        self.training_data = []
        
        # RL training parameters
        self.training_frequency = 10
        self.step_count = 0
    
    def solve(self, initial_problem: str, target_solutions: int = 1) -> Dict:
        """Solve problem with RL-enhanced adaptive reasoning"""
        
        # Initialize mathematical state from problem
        initial_math_state = MathematicalState()
        initial_math_state.goal = initial_problem
        
        # Extract equations from problem statement
        equations = self.generator._extract_equations_from_text(initial_problem)
        initial_math_state.equations.extend(equations)
        
        initial_state = ReasoningState(
            content=initial_problem,
            math_state=initial_math_state,
            score=1.0,
            confidence=0.8,
            depth=0
        )
        
        heapq.heappush(self.active_states, initial_state)
        self.backtrack_manager.add_state(initial_state)
        
        solutions_found = 0
        step_count = 0
        
        while (self.active_states or solutions_found < target_solutions) and step_count < self.max_depth * 2:
            step_count += 1
            self.step_count += 1
            
            # Check if backtracking is needed
            if not self.active_states or self.backtrack_manager.should_backtrack(self.active_states):
                backtrack_state = self.backtrack_manager.find_backtrack_point(self.active_states)
                if backtrack_state:
                    # Clear current states and restart from backtrack point
                    self.active_states.clear()
                    heapq.heappush(self.active_states, backtrack_state)
                elif not self.active_states:
                    break
            
            if not self.active_states:
                break
            
            # Get current best state
            current_state = heapq.heappop(self.active_states)
            
            # Add to history
            self.backtrack_manager.add_state(current_state)
            
            # Check if this is a solution
            if self._is_solution(current_state):
                current_state.final_answer = self.answer_extractor.extract_answer(current_state.content)
                self.completed_states.append(current_state)
                solutions_found += 1
                
                # Reward successful path
                self.scorer.add_experience(current_state, 1.0)
                continue
            
            # Check termination conditions
            if (current_state.score < self.min_score_threshold or 
                current_state.depth >= self.max_depth):
                
                current_state.is_dead_end = True
                current_state.failure_reason = self.backtrack_manager.analyze_failure(current_state)
                self.completed_states.append(current_state)
                
                # Negative reward for dead ends
                self.scorer.add_experience(current_state, -0.5)
                continue
            
            # Generate candidate continuations
            continuations = self.generator.generate_continuations(
                current_state, self.n_candidates
            )
            
            # Create new states
            candidate_states = []
            for continuation, new_math_state, step_type in continuations:
                new_content = current_state.content + " " + continuation
                new_state = ReasoningState(
                    content=new_content,
                    math_state=new_math_state,
                    depth=current_state.depth + 1,
                    parent_id=current_state.state_id,
                    step_type=step_type
                )
                candidate_states.append(new_state)
            
            # Score candidates
            score_confidence_pairs = self.scorer.score_states(candidate_states)
            for i, (score, confidence) in enumerate(score_confidence_pairs):
                candidate_states[i].score = score
                candidate_states[i].confidence = confidence
            
            # Sort and select survivors
            candidate_states.sort(key=lambda x: x.score, reverse=True)
            survivors = candidate_states[:self.k_survivors]
            pruned = candidate_states[self.k_survivors:]
            
            # Record pruning
            self.pruning_history.append({
                'step': step_count,
                'parent_state_id': current_state.state_id,
                'n_generated': len(candidate_states),
                'n_survivors': len(survivors),
                'n_pruned': len(pruned),
                'survivor_scores': [s.score for s in survivors],
                'pruned_scores': [s.score for s in pruned],
                'backtrack_triggered': False
            })
            
            # Add survivors to active queue
            for survivor in survivors:
                heapq.heappush(self.active_states, survivor)
            
            # Mark pruned states as dead ends
            for pruned_state in pruned:
                pruned_state.is_dead_end = True
                pruned_state.failure_reason = "Pruned due to low score"
                self.completed_states.append(pruned_state)
            
            # Add training experiences
            for survivor in survivors:
                reward = 0.1 if survivor.score > 0.5 else -0.1
                self.scorer.add_experience(current_state, reward, survivor)
            
            # Periodic RL training
            if self.step_count % self.training_frequency == 0:
                self.scorer.train_step()
        
        # Final cleanup
        while self.active_states:
            state = heapq.heappop(self.active_states)
            self.completed_states.append(state)
        
        return self._compile_results()
    
    def _is_solution(self, state: ReasoningState) -> bool:
        """Check if state represents a complete solution"""
        # Check if we have solved for all variables
        math_state = state.math_state
        
        # Simple heuristic: if we have variables with numeric values and high progress
        solved_vars = sum(1 for v in math_state.variables.values() 
                         if isinstance(v, (int, float)) or 
                         (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit()))
        
        progress = math_state.progress_indicators.get('goal_proximity', 0)
        
        # Also check for answer-like patterns in text
        has_answer_pattern = bool(re.search(r'(answer|result|solution).{0,20}(\d+)', 
                                          state.content.lower()))
        
        return (solved_vars > 0 and progress > 0.7) or has_answer_pattern or \
               ('=' in state.content and solved_vars > 0)
    
    def _compile_results(self) -> Dict:
        """Compile comprehensive results"""
        if not self.completed_states:
            return {"error": "No completed states found"}
        
        # Separate solutions from dead ends
        solutions = [s for s in self.completed_states if not s.is_dead_end or self._is_solution(s)]
        dead_ends = [s for s in self.completed_states if s.is_dead_end and not self._is_solution(s)]
        
        best_state = max(self.completed_states, key=lambda x: x.score) if self.completed_states else None
        
        # Extract final answer from best state
        final_answer = None
        if best_state:
            final_answer = best_state.final_answer or self.answer_extractor.extract_answer(best_state.content)
        
        # Failure analysis
        failure_analysis = {}
        for state in dead_ends:
            reason = state.failure_reason or "Unknown"
            failure_analysis[reason] = failure_analysis.get(reason, 0) + 1
        
        return {
            "best_solution": {
                "content": best_state.content if best_state else "No solution found",
                "score": best_state.score if best_state else 0,
                "confidence": best_state.confidence if best_state else 0,
                "depth": best_state.depth if best_state else 0,
                "final_answer": final_answer,
                "mathematical_state": {
                    "equations": [str(eq) for eq in best_state.math_state.equations] if best_state else [],
                    "variables": best_state.math_state.variables if best_state else {},
                    "applied_techniques": [t.value for t in best_state.math_state.applied_techniques] if best_state else []
                }
            },
            "all_solutions": [
                {
                    "content": s.content,
                    "score": s.score,
                    "confidence": s.confidence,
                    "final_answer": s.final_answer or self.answer_extractor.extract_answer(s.content),
                    "variables_solved": s.math_state.variables
                }
                for s in sorted(solutions, key=lambda x: x.score, reverse=True)[:5]
            ],
            "statistics": {
                "total_states_explored": len(self.completed_states),
                "solutions_found": len(solutions),
                "dead_ends": len(dead_ends),
                "backtrack_episodes": sum(1 for h in self.pruning_history if h.get('backtrack_triggered', False)),
                "rl_training_steps": self.scorer.training_step,
                "experience_buffer_size": self.scorer.experience_buffer.size()
            },
            "failure_analysis": failure_analysis,
            "suggested_improvements": self.backtrack_manager.suggest_alternative_strategy(dead_ends[:5])
        }

def main():
    """Main function to run experiments"""
    print("🚀 ARTEMIS: Adaptive Reinforcement Learning Mathematical Intelligence System")
    print("=" * 80)
    
    # Create experiment configuration
    config = ExperimentConfig(
        model_params={
            'feature_dim': 50,
            'n_candidates': 4,
            'k_survivors': 2,
            'max_depth': 8,
            'min_score_threshold': 0.2
        },
        training_params={
            'num_epochs': 5,
            'batch_size': 16,
            'learning_rate': 0.001,
            'training_frequency': 5
        },
        evaluation_params={
            'datasets': ['gsm8k'],
            'max_problems': 15,
            'compare_baselines': True
        },
        experiment_name="artemis_demo",
        output_dir="./artemis_results",
        use_wandb=False
    )
    
    # Run experiment
    runner = ExperimentRunner(config)
    results = runner.run_experiment()
    
    print("\n" + "="*80)
    print("🎯 EXPERIMENT COMPLETED!")
    print(f"📊 Final Accuracy: {results.accuracy:.3f}")
    print(f"🔬 Total Problems Solved: {results.efficiency_metrics.get('success_rate', 0) * config.evaluation_params['max_problems']:.0f}")
    print(f"📈 RL Training Steps: {results.training_metrics.get('loss', [0]).__len__()}")
    
    if results.baseline_comparisons:
        print("\n🏆 BASELINE COMPARISONS:")
        for baseline, metrics in results.baseline_comparisons.items():
            print(f"  {baseline}: {metrics.get('accuracy', 0):.3f} accuracy")
    
    print(f"\n📁 Results saved to: {config.output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()