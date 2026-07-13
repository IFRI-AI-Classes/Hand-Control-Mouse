"""
Ce fichier regarde la forme de la main et décide quel geste c'est
(par exemple : doigts pincés = clic).
Il dit juste "c'est un clic", il ne clique pas lui-même.
"""

# C'est supposé que les landmarks seront chargées depuis le fichier detection 

# Déjà le curseur est piloté par le bout de l'index (Landmark 8)

# Clic gauche sera le toucher entre l'index et le pouce

# Clic droit sera le toucher entre le pouce et le majeur

# Scroll piloté par le majeur et l'index. De base c'est le mode normal. Le système entre en mode scroll uniquement si l'index et le majeur sont levés et que les autres doigts sont repliés

from enum import Enum
import math



class Gesture(Enum):
    NONE = 0
    LEFT_CLICK = 1
    RIGHT_CLICK = 2
    SCROLL_UP = 3
    SCROLL_DOWN = 4
    DRAG = 5

class GestureRecognizer:
    def __init__(self , click_threshold = 0.04, scroll_threshold = 0.02 ):
        
        # Sensibilité du click
        self.click_threshold = click_threshold
        
        # Position précédente du majeur
        self.previous_middle_y = None
        
        # Sensibilité du scroll
        self.scroll_threshold = scroll_threshold
    
    def distance(self, point1, point2):
        """
        Calcule la distance euclidienne entre deux points.
        Chaque point est un tuple (x, y).
        """

        return math.hypot(
    point2[0] - point1[0],
    point2[1] - point1[1]
)

    def get_point(self, landmarks, index):
        """
        Récuperer les coordonnées d'un landmark
        """
        
        lm = landmarks[index]
        return (lm.x, lm.y)
    
    
    
    def _get_middle_delta(self, landmarks):
        """
        Pour trouver la distance entre la majeur de la frame precedente et de la frame actuelle
        """
        
        middle_y = landmarks[12].y

        if self.previous_middle_y is None:
            self.previous_middle_y = middle_y
            return 0

        delta = self.previous_middle_y - middle_y
        self.previous_middle_y = middle_y

        return delta

    def is_left_click(self, landmarks):
        """
        Vérifie si le mouvement effectué est un click gauche (index-8 et pouche-4 suivant le seuil threshold )
        """
        thumb = self.get_point(landmarks, 4)
        index = self.get_point(landmarks, 8)

        d = self.distance(thumb, index)

        return d < self.click_threshold
    

    def is_right_click(self, landmarks):
        """
        Vérifie si le mouvement effectué est un click droit (index-8 et majeur-12 suivant le seuil threshold )
        """
        thumb = self.get_point(landmarks, 4)
        middle = self.get_point(landmarks, 12)

        d = self.distance(thumb, middle)

        return d < self.click_threshold

    def is_scroll_up(self, delta):
        """
        Vérifie si c'est un scroll up.
        Calcule la distance entre le majeur de la frame précedente et celle de la frame actuelle.
        Pour un scroll up, y_new < y_prev => y_new - y_prev < 0
        """
        
        
        return delta > self.scroll_threshold
    
    
    def is_scroll_down(self , delta):
        """
        Vérifie si c'est un scroll down.
        Calcule la distance entre le majeur de la frame précedente et celle de la frame actuelle.
        Pour un scroll down, y_new > y_prev => y_new - y_prev > 0
        """        
        
        return delta < -self.scroll_threshold


    def _recognize_scroll(self, landmarks):
        """
        Fonction pour centraliser tous les types de scrolls (up ou Down)
        """
        

        delta = self._get_middle_delta(landmarks)

        if delta > self.scroll_threshold:
            return Gesture.SCROLL_UP

        if delta < -self.scroll_threshold:
            return Gesture.SCROLL_DOWN

        return Gesture.NONE
    
    
    def is_drag(self , landmarks):
        pass
    
    def recognize(self, landmarks):

        """
        Centraliser et detection du mouvement
        """
        
        if landmarks is None:
            return Gesture.NONE

        if self.is_left_click(landmarks):
            return Gesture.LEFT_CLICK

        if self.is_right_click(landmarks):
            return Gesture.RIGHT_CLICK

        scroll = self._recognize_scroll(landmarks)

        if scroll != Gesture.NONE:
            return scroll

        if self.is_drag(landmarks):
            return Gesture.DRAG

        return Gesture.NONE
