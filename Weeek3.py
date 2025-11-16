# Plant Disease Detection - Final 20%
# Disease Information Database + Streamlit Web Application

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import cv2
import os

# =====================================
# 1. DISEASE INFORMATION DATABASE
# =====================================

class DiseaseDatabase:
    """
    Comprehensive database of plant diseases with treatment recommendations
    """
    
    def __init__(self):
        self.diseases = {
            # TOMATO DISEASES
            "Tomato_Bacterial_spot": {
                "crop": "Tomato",
                "disease_name": "Bacterial Spot",
                "severity": "High",
                "description": "Bacterial spot causes dark, greasy-looking spots on leaves, stems, and fruit. Severe infections can lead to defoliation and reduced yields.",
                "symptoms": [
                    "Small, dark brown spots with yellow halos on leaves",
                    "Spots on fruit that are raised and have white halos",
                    "Defoliation in severe cases",
                    "Stunted plant growth"
                ],
                "causes": [
                    "Xanthomonas bacteria",
                    "Warm, humid weather conditions",
                    "Water splash from rain or irrigation",
                    "Infected seeds or transplants"
                ],
                "treatment": [
                    "Remove and destroy infected plants",
                    "Apply copper-based bactericides",
                    "Use disease-resistant varieties",
                    "Avoid overhead irrigation",
                    "Rotate crops with non-host plants",
                    "Disinfect tools between plants"
                ],
                "prevention": [
                    "Use certified disease-free seeds",
                    "Space plants properly for air circulation",
                    "Avoid working with plants when wet",
                    "Apply mulch to prevent soil splash"
                ],
                "organic_solutions": [
                    "Copper sulfate spray",
                    "Neem oil application",
                    "Baking soda solution (1 tbsp per gallon water)"
                ]
            },
            
            "Tomato_Early_blight": {
                "crop": "Tomato",
                "disease_name": "Early Blight",
                "severity": "Medium",
                "description": "Early blight is a common fungal disease affecting tomatoes. It causes distinctive target-like lesions on leaves and can significantly reduce yield.",
                "symptoms": [
                    "Brown spots with concentric rings (target-like)",
                    "Yellow halo around spots",
                    "Lower leaves affected first",
                    "Premature leaf drop"
                ],
                "causes": [
                    "Alternaria solani fungus",
                    "Warm temperatures (75-85°F)",
                    "High humidity",
                    "Poor air circulation"
                ],
                "treatment": [
                    "Remove infected lower leaves",
                    "Apply fungicides (chlorothalonil, mancozeb)",
                    "Improve air circulation",
                    "Water at soil level only",
                    "Apply organic fungicides like copper"
                ],
                "prevention": [
                    "Rotate crops (3-4 year cycle)",
                    "Stake or cage plants for better airflow",
                    "Mulch around plants",
                    "Avoid overhead watering",
                    "Space plants adequately"
                ],
                "organic_solutions": [
                    "Compost tea spray",
                    "Copper fungicide",
                    "Baking soda spray"
                ]
            },
            
            "Tomato_Late_blight": {
                "crop": "Tomato",
                "disease_name": "Late Blight",
                "severity": "Critical",
                "description": "Late blight is a devastating disease that can destroy entire crops within days. It's the same disease that caused the Irish potato famine.",
                "symptoms": [
                    "Water-soaked spots on leaves",
                    "White fungal growth on undersides",
                    "Brown lesions on stems",
                    "Fruit rot with greasy appearance",
                    "Rapid plant death"
                ],
                "causes": [
                    "Phytophthora infestans",
                    "Cool, wet weather",
                    "High humidity (>90%)",
                    "Temperatures 50-70°F"
                ],
                "treatment": [
                    "IMMEDIATE removal of infected plants",
                    "Apply fungicides preventively",
                    "Destroy all infected material (burn or bury)",
                    "Use systemic fungicides (metalaxyl)",
                    "DO NOT compost infected plants"
                ],
                "prevention": [
                    "Use resistant varieties",
                    "Ensure excellent air circulation",
                    "Water in morning only",
                    "Monitor weather forecasts",
                    "Apply preventive fungicides in wet weather"
                ],
                "organic_solutions": [
                    "Copper-based fungicides (apply before infection)",
                    "Remove plants at first sign",
                    "Prevention is key - organic treatment difficult"
                ]
            },
            
            "Tomato_Leaf_Mold": {
                "crop": "Tomato",
                "disease_name": "Leaf Mold",
                "severity": "Medium",
                "description": "Leaf mold thrives in greenhouse conditions with high humidity. It rarely affects outdoor plants.",
                "symptoms": [
                    "Pale green to yellow spots on upper leaf surface",
                    "Olive-green to brown fuzzy growth underneath",
                    "Leaves curl and die",
                    "Reduced photosynthesis"
                ],
                "causes": [
                    "Passalora fulva fungus",
                    "High humidity (>85%)",
                    "Poor air circulation",
                    "Greenhouse conditions"
                ],
                "treatment": [
                    "Reduce humidity below 85%",
                    "Improve ventilation",
                    "Remove infected leaves",
                    "Apply fungicides (chlorothalonil)",
                    "Space plants properly"
                ],
                "prevention": [
                    "Use resistant varieties",
                    "Ensure good air flow",
                    "Avoid overhead irrigation",
                    "Prune lower leaves",
                    "Use dehumidifiers in greenhouses"
                ],
                "organic_solutions": [
                    "Increase air circulation",
                    "Reduce humidity",
                    "Copper sprays"
                ]
            },
            
            "Tomato_Septoria_leaf_spot": {
                "crop": "Tomato",
                "disease_name": "Septoria Leaf Spot",
                "severity": "Medium",
                "description": "A common fungal disease that affects tomato leaves, causing small circular spots with gray centers.",
                "symptoms": [
                    "Small circular spots with dark borders",
                    "Gray or tan centers",
                    "Tiny black dots in center (fungal spores)",
                    "Starts on lower leaves"
                ],
                "causes": [
                    "Septoria lycopersici fungus",
                    "Wet weather",
                    "Water splash",
                    "High humidity"
                ],
                "treatment": [
                    "Remove infected leaves",
                    "Apply fungicides (chlorothalonil, copper)",
                    "Mulch to prevent soil splash",
                    "Water at base of plants only"
                ],
                "prevention": [
                    "Crop rotation",
                    "Stake plants for airflow",
                    "Remove plant debris",
                    "Avoid overhead watering"
                ],
                "organic_solutions": [
                    "Copper fungicide",
                    "Neem oil",
                    "Remove affected leaves promptly"
                ]
            },
            
            "Tomato_Spider_mites_Two_spotted_spider_mite": {
                "crop": "Tomato",
                "disease_name": "Two-Spotted Spider Mite",
                "severity": "Medium",
                "description": "Tiny arachnids that suck plant juices, causing stippling and webbing on leaves.",
                "symptoms": [
                    "Yellow stippling on leaves",
                    "Fine webbing on undersides",
                    "Bronzed or silvery appearance",
                    "Leaf drop in severe cases"
                ],
                "causes": [
                    "Spider mites (Tetranychus urticae)",
                    "Hot, dry conditions",
                    "Dusty environments",
                    "Water stress"
                ],
                "treatment": [
                    "Spray with strong water jet",
                    "Apply insecticidal soap",
                    "Use miticides if severe",
                    "Increase humidity around plants",
                    "Release predatory mites"
                ],
                "prevention": [
                    "Regular water spraying",
                    "Maintain plant health",
                    "Avoid water stress",
                    "Encourage beneficial insects"
                ],
                "organic_solutions": [
                    "Neem oil spray",
                    "Insecticidal soap",
                    "Predatory mites (Phytoseiulus persimilis)",
                    "Garlic spray"
                ]
            },
            
            "Tomato_Target_Spot": {
                "crop": "Tomato",
                "disease_name": "Target Spot",
                "severity": "Medium",
                "description": "Fungal disease causing concentric ring patterns on leaves, similar to early blight.",
                "symptoms": [
                    "Brown spots with concentric rings",
                    "Affects leaves, stems, and fruit",
                    "Defoliation in severe cases"
                ],
                "causes": [
                    "Corynespora cassiicola fungus",
                    "Warm, humid conditions",
                    "Poor air circulation"
                ],
                "treatment": [
                    "Apply fungicides",
                    "Remove infected plant parts",
                    "Improve air circulation",
                    "Reduce humidity"
                ],
                "prevention": [
                    "Space plants properly",
                    "Avoid overhead irrigation",
                    "Crop rotation",
                    "Use disease-free seeds"
                ],
                "organic_solutions": [
                    "Copper fungicides",
                    "Neem oil",
                    "Compost tea"
                ]
            },
            
            "Tomato_Tomato_Yellow_Leaf_Curl_Virus": {
                "crop": "Tomato",
                "disease_name": "Yellow Leaf Curl Virus",
                "severity": "Critical",
                "description": "Viral disease transmitted by whiteflies causing severe yield loss.",
                "symptoms": [
                    "Upward curling of leaves",
                    "Yellowing between veins",
                    "Stunted growth",
                    "Reduced fruit production",
                    "Small, deformed leaves"
                ],
                "causes": [
                    "Begomovirus",
                    "Whitefly transmission",
                    "Infected plant material"
                ],
                "treatment": [
                    "NO CURE - remove infected plants",
                    "Control whitefly population",
                    "Use insecticides for whiteflies",
                    "Remove and destroy infected plants"
                ],
                "prevention": [
                    "Use resistant varieties",
                    "Control whiteflies with yellow sticky traps",
                    "Use insect screening in greenhouses",
                    "Remove weeds (alternate hosts)",
                    "Use virus-free transplants"
                ],
                "organic_solutions": [
                    "Neem oil for whitefly control",
                    "Yellow sticky traps",
                    "Reflective mulches",
                    "Remove infected plants immediately"
                ]
            },
            
            "Tomato_Tomato_mosaic_virus": {
                "crop": "Tomato",
                "disease_name": "Tomato Mosaic Virus",
                "severity": "High",
                "description": "Highly contagious viral disease causing mottled leaves and reduced yields.",
                "symptoms": [
                    "Mottled light and dark green on leaves",
                    "Distorted, fern-like leaves",
                    "Stunted growth",
                    "Reduced fruit quality",
                    "Internal browning of fruit"
                ],
                "causes": [
                    "Tobamovirus",
                    "Mechanical transmission",
                    "Infected tools and hands",
                    "Contaminated seeds"
                ],
                "treatment": [
                    "NO CURE - remove infected plants",
                    "Disinfect all tools",
                    "Wash hands thoroughly",
                    "Avoid tobacco use near plants"
                ],
                "prevention": [
                    "Use resistant varieties",
                    "Purchase certified virus-free seeds",
                    "Disinfect tools with 10% bleach",
                    "Don't smoke near plants",
                    "Remove infected plants immediately"
                ],
                "organic_solutions": [
                    "Prevention only - no treatment exists",
                    "Strict sanitation practices",
                    "Quarantine new plants"
                ]
            },
            
            "Tomato_healthy": {
                "crop": "Tomato",
                "disease_name": "Healthy Plant",
                "severity": "None",
                "description": "Your tomato plant appears healthy! Continue with good care practices.",
                "symptoms": [
                    "Deep green foliage",
                    "No spots or discoloration",
                    "Vigorous growth",
                    "No wilting"
                ],
                "causes": ["N/A - Plant is healthy"],
                "treatment": ["Continue regular care"],
                "prevention": [
                    "Maintain consistent watering",
                    "Ensure adequate nutrition",
                    "Monitor for pests regularly",
                    "Provide proper support",
                    "Maintain good air circulation"
                ],
                "organic_solutions": [
                    "Compost tea for nutrition",
                    "Companion planting (basil, marigolds)",
                    "Regular inspection"
                ]
            },
            
            # POTATO DISEASES
            "Potato_Early_blight": {
                "crop": "Potato",
                "disease_name": "Early Blight",
                "severity": "Medium",
                "description": "Common fungal disease of potatoes causing target-like lesions on leaves.",
                "symptoms": [
                    "Brown lesions with concentric rings",
                    "Yellow halo around spots",
                    "Lower leaves affected first",
                    "Stem lesions"
                ],
                "causes": [
                    "Alternaria solani",
                    "Warm, humid weather",
                    "Plant stress"
                ],
                "treatment": [
                    "Apply fungicides",
                    "Remove infected foliage",
                    "Improve drainage",
                    "Reduce humidity"
                ],
                "prevention": [
                    "Crop rotation (3 years)",
                    "Use certified seed potatoes",
                    "Maintain plant vigor",
                    "Proper spacing"
                ],
                "organic_solutions": [
                    "Copper fungicides",
                    "Compost tea",
                    "Proper crop rotation"
                ]
            },
            
            "Potato_Late_blight": {
                "crop": "Potato",
                "disease_name": "Late Blight",
                "severity": "Critical",
                "description": "The most destructive potato disease. Can destroy entire fields in days.",
                "symptoms": [
                    "Water-soaked lesions on leaves",
                    "White fungal growth on undersides",
                    "Blackened stems",
                    "Brown, rotten tubers",
                    "Rapid plant collapse"
                ],
                "causes": [
                    "Phytophthora infestans",
                    "Cool, wet conditions",
                    "High humidity"
                ],
                "treatment": [
                    "Destroy infected plants immediately",
                    "Apply systemic fungicides",
                    "Harvest before disease reaches tubers",
                    "Do not store infected tubers"
                ],
                "prevention": [
                    "Use resistant varieties",
                    "Plant certified seed",
                    "Hill plants properly",
                    "Fungicide application in wet weather",
                    "Monitor weather forecasts"
                ],
                "organic_solutions": [
                    "Copper fungicides (preventive only)",
                    "Immediate removal of infected plants",
                    "Prevention is critical"
                ]
            },
            
            "Potato_healthy": {
                "crop": "Potato",
                "disease_name": "Healthy Plant",
                "severity": "None",
                "description": "Your potato plant is healthy! Maintain good growing practices.",
                "symptoms": [
                    "Green, vigorous foliage",
                    "No lesions or spots",
                    "Strong stems"
                ],
                "causes": ["N/A - Plant is healthy"],
                "treatment": ["Continue regular care"],
                "prevention": [
                    "Proper hilling",
                    "Consistent watering",
                    "Regular inspection",
                    "Harvest at right time"
                ],
                "organic_solutions": [
                    "Compost application",
                    "Companion planting",
                    "Regular monitoring"
                ]
            }
        }
    
    def get_disease_info(self, disease_name):
        """Get complete information for a disease"""
   
