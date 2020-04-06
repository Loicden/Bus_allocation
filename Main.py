# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:52:02 2020
@author: vinch
A faire: 
"""

import random as rd
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import pylab

#Hyperparamètres
v = 10                  # Vitesse moyenne des bus (m/s)
f = 6/3600              # Fréquence des bus (bus/s)
c = 20              # le temps nécessaire pour changer de bus (correspondance, en s)

taille_map = 1000     # (x,y) app à [-taille_map,+taille_map]²
distance_minimale = 150 # Pour éviter que des arrêts ne soient trop prêts
hypA = (20,30) #Pas dépasser 35-40 l'algo est de complexité quadratique ça fait bobo
hypL = (3,4)
hypT = (2,30)
nb_lignes_max = 3


##### Définitions des classes et fonctions #####

def ligneInit(Graphe,depart) :
    """ Renvoie la première ligne du tableau """
    L = []
    # nombre de lignes de Graphe donc nombre de sommets
    n = len(Graphe)
    for j in range(n) :
        poids = Graphe[depart][j]
        if poids :
            # si l’arête est présente
            L.append([ poids, depart ])
        else :
            L.append(False)
    return [L]


def SommetSuivant(T, S_marques) :
    """ En considérant un tableau et un ensemble de sommets marqués,
    détermine le prochain sommet marqué. """
    L = T[-1]
    n = len(L)
    # minimum des longueurs, initialisation
    min = False
    #print("ERROR",S_marques)
    for i in range(n) :
        if not(i in S_marques) :
            # si le sommet d’indice i n’est pas marqué
            if L[i]:
                if not(min) or L[i][0] < min :
                    # on trouve un nouveau minimum
                    # ou si le minimum n’est pas défini
                    min = L[i][0]
                    marque = i
        
    return (marque)
    
    
def ajout_ligne(T,S_marques,Graphe) :
    """ Ajoute une ligne supplémentaire au tableau """
    L = T[-1]
    n = len(L)
    # La prochaine ligne est une copie de la précédente,
    # dont on va modifier quelques valeurs.
    Lnew = L.copy()
    # sommet dont on va étudier les voisins
    S = S_marques[-1]
    # la longueur du (plus court) chemin associé
    long = L[S][0]
    for j in range(n) :
        if j not in S_marques:
            poids = Graphe[S][j]
            if poids :
                # si l’arète (S,j) est présente
                if not(L[j]) : # L[j] = False
                    Lnew[j] = [ long + poids, S ]
                else :
                    if long + poids < L[j][0] :
                        Lnew[j] = [ long + poids, S ]
    T.append(Lnew)
    # Calcul du prochain sommet marqué
    S_marques.append(SommetSuivant(T, S_marques))
    return T, S_marques


def calcule_tableau(Graphe, depart) :
    """ Calcule le tableau de l’algorithme de Dijkstra """
    n = len(Graphe)
    # Initialisation de la première ligne du tableau
    # Avec ces valeurs, le premier appel à ajout_ligne
    # fera le vrai travail d’initialisation
    T=[[False] *n]
    T[0][depart] = [depart, 0]
    
    # liste de sommets marques
    S_marques = [ depart ]
    
    while len(S_marques) < n :
        T, S_marques = ajout_ligne(T, S_marques, Graphe)
        
    return T


def plus_court_chemin(Graphe, depart, arrivee) :
    """ Détermine le plus court chemin entre depart et arrivee dans
    le Graphe"""
    #n = len(Graphe)
    # calcul du tableau de Dijkstra
    Truc = calcule_tableau (Graphe,depart)
    # liste qui contiendra le chemin le plus court, on place l’arrivée
    C = [ arrivee ]
    while C[-1] != depart :
        C.append( Truc[-1][ C[-1] ][1] )
        # Renverse C, pour qu’elle soit plus lisible
    C.reverse()
    return C
    
class Arret:
    """Objet Arret qui contient ses coordonnées, si c'est la station principale ou non
    Une fonction de description __str__
    Un mutateur pour la déclarer comme gare principale
    Un mutateur pour modifier ses coordonnées"""
    def __init__(self,num):
        self.x = rd.uniform(-taille_map,taille_map)
        self.y = rd.uniform(-taille_map,taille_map)
        self.r_m = False              # Par défaut, pas la gare centrale
        self.num = num
        self.ligne = 0
        
    def __str__(self):
        return "(Arret : X = " + str(self.x) + ' ; Y = '+ str(self.y) + ")"
    
    def set_Gare_Centrale(self):
        self.r_m = True
        
    def set_coor (self,x,y):
        self.x = x
        self.y = y


class Ligne:
    """ Objet ligne qui comporte l'ensemble des arrêts, leur nombre ainsi que 
    la matrice des distances euclidiennes
    Un fonction descriptive
    Un mutateur pour ajouter un arrêt
    Un mutateur pour supprimer un arrêt
    Une fonction qui ordonne les arrêts"""
    def __init__(self, arrets_ligne,num):
        self.arrets = arrets_ligne
        self.Nb_arrets = len(self.arrets)
        self.Tab_dist = self.D_build()    # Tableau des distances entre chaque arrêts
        self.num = num
        
    def __str__(self):
        print("Ligne -", self.Nb_arrets, 'arrêts')
        print("Ses arrêts sont ")
        for arret in self.arrets :
            if arret.r_m == True :
                print(arret, "(Gare Centrale)")
            else :
                print(arret)
        return ""
    
    def ajout_arret(self, arret):
        self.arrets.append(arret)
        self.Nb_arrets = len(self.arrets)
        print("On vient d'ajouter l'arrêt", arret, "à cette ligne.")

    def suppr_arret(self, arret):
        if arret not in self.arrets :
            print(arret,"n'est pas dans les arrêts de cette ligne.")
        else :
            self.arrets.remove(arret)
            self.Nb_arrets = len(self.arrets)
            #print("On vient de supprimer l'arrêt", arret, "de cette ligne.")    
            
    def D_build(self):
        """
        Construction de la matrice des distance euclidiennes entre arrêts
        """
        nb_arrets = len(self.arrets)
        self.D = np.zeros([nb_arrets, nb_arrets])
        for i in range(nb_arrets):
            for j in range(i, nb_arrets):
                self.D[i,j] = np.sqrt(abs(self.arrets[i].x - self.arrets[j].x)**2 + abs(self.arrets[i].y - self.arrets[j].y)**2)
                self.D[j,i] = self.D[i,j]
        self.D = np.array(self.D).tolist()
        return self.D
    
    def ordonner_arrets(self):
        """Fonction qui ordonnes les arrêts de la manière suivante:
        Choisit un point de départ au hasard. 
        A chaque itération va trouver la station la plus proche du dernier
        élément de la liste.
        A la fin mesure la longueur totale afin de garder la ligne la plus courte
        Donc en somme cette fonctionne ordonne les arrêts afin de minimiser
        la distance parcourue par le bus.
        """
        Distance_totale = 100000
        Ordre = []
        #print()
        #for i in self.Tab_dist:
        #    print(i)
        #print()
        for Pt_depart in range(1,self.Nb_arrets+1):
            #print("On commence avec", Pt_depart)
            Element = [Pt_depart]
            Tab_dist_MST = []
            #print()
            while len(Element) != self.Nb_arrets:
                Distance_minimale = 10000
                i = Element[-1]
                #print()
                #print("On se place sur le point", i)
                for j in self.Tab_dist[i-1]:
                    #print("On considère la distance", j)
                    if j < Distance_minimale and j != 0 and self.Tab_dist[i-1].index(j)+1 not in Element:
                        Distance_minimale = j
                        Plus_proche_voisin = self.Tab_dist[i-1].index(j)+1
                        #print("Benef, le voisin est :", self.Tab_dist[i-1].index(j)+1)
                    else :
                        #print("Pas benef")
                        pass
                Element.append(Plus_proche_voisin)
                #print("Element :", Element)
                Tab_dist_MST.append(Distance_minimale)
                #print(Tab_dist_MST)
                #print("La distance est de :", sum(Tab_dist_MST))
            #print("On a la liste d'idx :", Element, "pour une distance totale de",sum(Tab_dist_MST))
            
            if sum(Tab_dist_MST) < Distance_totale :
                #print("On remplace")
                Distance_totale = sum(Tab_dist_MST)
                Ordre = Element
        
        arrets_ord = []
        for i in Ordre:
            arrets_ord.append(self.arrets[i-1])

        self.arrets = arrets_ord

##### Affichage #####

def display_arrets(arrets):
    X = [arrets[i].x for i in range(len(arrets))]
    Y = [arrets[i].y for i in range(len(arrets))]
    plt.plot(X,Y, 'or')
    for i in range(len(arrets)):
        plt.annotate(arrets[i].num,(arrets[i].x,arrets[i].y))
    
def display_lignes(lignes):
    for l in lignes:
        X = [a.x for a in l.arrets]
        Y = [a.y for a in l.arrets]
        plt.plot(X,Y)

def create_arrets():
    
    nb_arrets = rd.randint(hypA[0],hypA[1])
    #print("On choisi de créer",nb_arrets,"arrêts.")
    #Création aléatoire d'un certain nombre d'arrêts
    arrets_list = []
    k = 0
    for i in range(nb_arrets):
        arret = Arret(i)
        while distance_mini(arret,arrets_list) < distance_minimale and k <100:
            k+=1
            arret = Arret(i)
        if k>=100:
            print("Contraintes trop fortes pour arrêts")
            return None
        arrets_list.append(arret)
    print(nb_arrets," arrêts générés")
    return arrets_list


class Reseau:

    def __init__(self, arrets_list):
        self.nb_arrets = len(arrets_list)
        self.arrets = copy.deepcopy(arrets_list)
        arrets_dispo = []
        for a in self.arrets:
            arrets_dispo.append(a)
            
        self.r_m = rd.choice(self.arrets)     # Arrêt principal
        self.r_m.set_Gare_Centrale()
        
        ##### Construction des tableaux et constantes #####
        
        self.D_build()         # Matrice des distances entre les arrêts i et j
        #print("D :", D)
        #print("T :", T)
        
        self.nb_lignes = rd.randint(hypL[0],hypL[1])     # On veut au moins deux arrêts par lignes, sachant que toutes doivent passer par la gare centrale
        #print("On choisi de créer",self.nb_lignes,"lignes.")
        self.lignes = []
        arrets_dispo.remove(self.r_m)
        nb_arrets_dispo = len(arrets_dispo)
        
        for i in range(1,self.nb_lignes+1):
#            print()
#            print("--- Construction d'une ligne ---")
            arrets_ligne_temp = [self.r_m]  # On ajoute forcement la gare centrale
            if i != self.nb_lignes :
                nb_arrets_ligne = rd.randint(1,nb_arrets_dispo-(self.nb_lignes-i))
                for j in range(nb_arrets_ligne):
                    arret = rd.choice(arrets_dispo)
                    arrets_dispo.remove(arret)
                    arrets_ligne_temp.append(arret)
            else : 
                arrets_ligne_temp += arrets_dispo         # On fait la dernière ligne avec les arrêts restants
                
            self.lignes.append(Ligne(arrets_ligne_temp,i-1))
            self.set_lignes_in_arrets()
#            print("La ligne créée est :", self.lignes[i-1])
            nb_arrets_dispo = len(arrets_dispo)
            self.lignes[i-1].ordonner_arrets()
#            print("Après arrangement, on a la ligne :", self.lignes[i-1])
        self.D_build()
        self.U_build()
        
    
    def distance_mini(self,arret):
        mini = taille_map*4
        for a in self.arrets:
            A = a.x - arret.x
            B = a.y - arret.y
            norme = np.sqrt(A**2 + B**2)   
            if norme < mini:
                mini = norme
        return mini
    
    def set_lignes_in_arrets(self):
        for l in self.lignes:
            for a in l.arrets:
                a.ligne = l
    
    def display(self,show):
        display_arrets(self.arrets)
        display_lignes(self.lignes)
        plt.grid()
        if show:
            plt.show()

    def D_build(self):
        """
        Construction de la matrice des distance euclidiennes entre arrêts
        """
        nb_arrets = len(self.arrets)
        self.D = np.zeros([nb_arrets, nb_arrets])
        for i in range(nb_arrets):
            for j in range(i, nb_arrets):
                self.D[i,j] = np.sqrt(abs(self.arrets[i].x - self.arrets[j].x)**2 + abs(self.arrets[i].y - self.arrets[j].y)**2)
                self.D[j,i] = self.D[i,j]
        self.D = np.array(self.D).tolist()
        return self.D

    def change_de_ligne(self,arc):
        if self.arrets[arc[1]].r_m and (self.arrets[arc[0]].ligne != self.arrets[arc[2]].ligne):
            #On arrive à la station principale, changement potentiel
            #Et il y a changement :o
            return True
        return False

    def distance_arret_initial(self, arret_num, next_arret_num):
        d_tot = 0
        arret = self.arrets[arret_num]
        next_arret = self.arrets[next_arret_num]
        ligne = next_arret.ligne
        if arret in ligne.arrets:
            index = ligne.arrets.index(arret)
        else:
            print("ligne ",ligne.num)
            print("arret ",arret.num," r_m ",arret.r_m)
            print("ligne_arrets ",[a.num for a in ligne.arrets])
        try:
            next_index = ligne.arrets.index(next_arret)
        except:
            print("ligne ",ligne.num)
            print("arret ",next_arret.num," r_m ",next_arret.r_m)
            print("ligne_arrets ",[a.num for a in ligne.arrets])
        if index - next_index > 0:
            # Alors on va en ordre croissant donc le premier arrêt est en 0
            for i in range(index):
                d_tot += self.D[ligne.arrets[i].num][ligne.arrets[i+1].num]
            return d_tot
        elif index - next_index < 0:
            # On va dans l'autre sens, le premier arrêt est le dernier
            for i in range(-1,index-len(ligne.arrets),-1):
                d_tot += self.D[ligne.arrets[i].num][ligne.arrets[i-1].num]
            return d_tot
        else:
            print("CRITICAL ERROR")
            return None

    def calcul_tps_trajet(self,chemin):
        tps = 0
        for i in range(len(chemin)-2):
            tps += self.D[chemin[i]][chemin[i+1]]/v
            # t = d/v
            if self.change_de_ligne(chemin[i:i+3]):
                tps += c
                #Il y a un paquet d'amélioration à faire.
                d_tot = self.distance_arret_initial(chemin[i+1],chemin[i+2])
                delta_t = d_tot/v - int(tps*f)/f - tps%(1/f)
                #value = 1/f - tps%(1/f) #Simplification du modèle ici, autant de bus que d'arrêts
                tps += delta_t
                
        tps += self.D[chemin[-2]][chemin[-1]]/v
        return tps
        
    def U_build(self):
        Graphe = [[False for i in range(len(self.arrets))] for j in range(len(self.arrets))]

        for l in self.lignes: #Pour chaque ligne
            
            for i in range(len(l.arrets)-1):
                Graphe[l.arrets[i].num][l.arrets[i+1].num] = self.D[l.arrets[i].num][l.arrets[i+1].num]
                Graphe[l.arrets[i+1].num][l.arrets[i].num] = self.D[l.arrets[i].num][l.arrets[i+1].num]
        self.U = np.zeros((len(Graphe),len(Graphe)))
        for i in range(len(self.arrets)):
            for j in range(len(self.arrets)):
                if j != i:
                    chemin = plus_court_chemin(Graphe,i,j)
                    self.U[i,j] = self.calcul_tps_trajet(chemin)
        self.U = np.array(self.U)
        #print(self.U)
        
        return self.U
        #U est maintenant la matrice qui contient les plus petites distances
        #entre les arrêts i vers j
        #Il faut encore prendre en compte le temps de changement d'arrêt.
        
    def calcul_ATT(self, T):
        sum0 = 0
        for x in range(len(liste_arrets)):
            sum1 = 0
            sum2 = 0
            for y in range(len(liste_arrets)):
                sum1 += self.U[x,y]*T[x,y]
                sum2 += T[x,y]
            sum0 += sum1/sum2
        self.ATT = sum0/len(liste_arrets)
        return self.ATT
    
    def petite_mut(self):
        
        mutation = False
        for i in self.lignes:               
            if i.Nb_arrets > 2 :  
                      # On supprime un arrêt au hasard, qui n'et pas la gare centrale.7
                arret_supp = rd.choice(i.arrets)
                while arret_supp.r_m == True :
                    arret_supp = rd.choice(i.arrets)
                i.suppr_arret(arret_supp)
                ligne_random = rd.choice(self.lignes)
                while ligne_random == i:
                    ligne_random = rd.choice(self.lignes)
                self.lignes[self.lignes.index(ligne_random)].arrets.append(arret_supp)
                #On rajoute cet arrêt à une autre ligne
            mutation = True
            break # ?
        
        if mutation == False:
            ligne_supp = rd.choice(self.lignes)
            arrets_supp = ligne_supp.arrets
            self.lignes.remove(ligne_supp)
            for i in arrets_supp :
                lig = rd.choice(self.lignes)
                lig.arrets.append(i)
            
        
    def grosse_mut(self, arrets_list):

        
        self.nb_arrets = len(arrets_list)
        self.arrets = copy.deepcopy(arrets_list)
        arrets_dispo = []
        for a in self.arrets:
            arrets_dispo.append(a)
            
        self.r_m = rd.choice(self.arrets)     # Arrêt principal
        self.r_m.set_Gare_Centrale()
        
        ##### Construction des tableaux et constantes #####
        
        self.D_build()         # Matrice des distances entre les arrêts i et j
        #print("D :", D)
        #print("T :", T)
        
        self.nb_lignes = rd.randint(hypL[0],hypL[1])     # On veut au moins deux arrêts par lignes, sachant que toutes doivent passer par la gare centrale
        #print("On choisi de créer",self.nb_lignes,"lignes.")
        self.lignes = []
        arrets_dispo.remove(self.r_m)
        nb_arrets_dispo = len(arrets_dispo)
        
        for i in range(1,self.nb_lignes+1):
#            print()
#            print("--- Construction d'une ligne ---")
            arrets_ligne_temp = [self.r_m]  # On ajoute forcement la gare centrale
            if i != self.nb_lignes :
                nb_arrets_ligne = rd.randint(1,nb_arrets_dispo-(self.nb_lignes-i))
                for j in range(nb_arrets_ligne):
                    arret = rd.choice(arrets_dispo)
                    arrets_dispo.remove(arret)
                    arrets_ligne_temp.append(arret)
            else : 
                arrets_ligne_temp += arrets_dispo         # On fait la dernière ligne avec les arrêts restants
                
            self.lignes.append(Ligne(arrets_ligne_temp,i-1))
            self.set_lignes_in_arrets()
#            print("La ligne créée est :", self.lignes[i-1])
            nb_arrets_dispo = len(arrets_dispo)
            self.lignes[i-1].ordonner_arrets()
#            print("Après arrangement, on a la ligne :", self.lignes[i-1])
        
    def reproduction(self, reseau_modele):
        
        ligne_heritee = rd.choice(reseau_modele.lignes)
        new_arrets_liste = copy.deepcopy(self.arrets)
        #print(self.arrets)
        for i in ligne_heritee.arrets:
            print(i)
            if i.r_m == False :
                print("delete")
                new_arrets_liste.remove(i)
        self.grosse_mut(new_arrets_liste)
        
        
def distance_mini(arret,arrets_liste):
    mini = taille_map*4
    for a in arrets_liste:
        A = a.x - arret.x
        B = a.y - arret.y
        norme = np.sqrt(A**2 + B**2)   
        if norme < mini:
            mini = norme
    return mini

def T_build(arrets):
    """
    Construction d'une matrice aléatoire qui représente le nombre de personnnes
    présentes à chaque arrêt
    """
    nb_arrets = len(arrets)
    T = np.zeros([nb_arrets, nb_arrets])
    for i in range(nb_arrets):
        for j in range(nb_arrets):
            if i != j :
                T[i,j] = int(rd.uniform(hypT[0],hypT[1]))
    return T

def ATT_liste_build(liste_reseaux, T):
    ATT_liste = []
    for i in liste_reseaux:
        i.calcul_ATT(T)
        ATT_liste.append(i.ATT)
    return ATT_liste

def Ordonner_reseaux(liste_reseaux, ATT_liste):
    ATT_liste_ord = copy.deepcopy(ATT_liste)
    ATT_liste_ord.sort()
    liste_reseaux_ord = []
    for i in ATT_liste_ord:
        liste_reseaux_ord.append(liste_reseaux[ATT_liste.index(i)])
    liste_reseaux = liste_reseaux_ord
    return liste_reseaux, ATT_liste_ord

def Optimisation(nb_iter, N_pop, p_M, p_m, p_s, liste_reseaux, T):
    
    p_best = 1 - (p_M + p_m + p_s)
    nb_M = int(np.floor(N_pop*p_M))
    nb_m = int(np.floor(N_pop*p_m))
    #nb_s = int(np.floor(N_pop*p_s))
    nb_best = int(np.floor(N_pop*p_best))
    if nb_best == 0:
        nb_best = 1
    
    itera = 0
    tab_iter = []
    tab_iter.append(itera)

    ATT_liste = ATT_liste_build(liste_reseaux, T)
    liste_reseaux, ATT_liste = Ordonner_reseaux(liste_reseaux, ATT_liste)
    perf = ATT_liste[0] #
    perf_tab = []
    perf_tab.append(perf)
    tps_init = time.time()
    
    while itera < nb_iter:
        print(itera)
        print(ATT_liste[0])
        itera += 1
        #plt.figure(figsize=(20,10))
        for i in range(N_pop):
            
            #plt.subplot(4,4,i+1)
            #liste_reseaux[i].display(False)
            #plt.title(ATT_liste[i])
            
            # On garde les meilleurs :
            if i <= nb_best:
                pass
            
            # On fait les petites mutations :
            elif i <= (nb_best + nb_M):
                liste_reseaux[i].petite_mut() # boucle infinie parfois 
                
            # On fait les grandes mutations :
            elif i <= (nb_best + nb_M + nb_m):
                liste_reseaux[i].grosse_mut(liste_arrets)
            # On fait le sexe :
            else :
                #liste_reseaux[i].reproduction(liste_reseaux[0])
                liste_reseaux[i].grosse_mut(liste_arrets)
            
            liste_reseaux[i].U_build()
        #plt.show()
        ATT_liste = ATT_liste_build(liste_reseaux, T)
        liste_reseaux, ATT_liste = Ordonner_reseaux(liste_reseaux, ATT_liste)
        perf = ATT_liste[0]
        perf_tab.append(perf)
        tab_iter.append(tab_iter[-1]+1)
        
    tps_final = time.time()
    Temps = tps_final-tps_init
    
    plt.figure(figsize=(20,10))
    plt.title("Meilleur individu")
    liste_reseaux[0].display(True)
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    line, = ax.plot(tab_iter,perf_tab,'bo', lw=2)
    #ax.set_yscale('log')
    pylab.show()
    
    return Temps

### MAIN ###

liste_arrets = create_arrets()
   
plt.close('all')
print("Figures fermées")
T = T_build(liste_arrets)
liste_reseaux = []
print("T construite")

#res = Reseau(liste_arrets)
#
#res.calcul_ATT(T)
#res.display(True)

N_pop = 16

for i in range(N_pop):
    print("individu ",i," créé")
    liste_reseaux.append(Reseau(liste_arrets))
    liste_reseaux[-1].D_build()
    liste_reseaux[-1].U_build()
    liste_reseaux[-1].calcul_ATT(T)
    #liste_reseaux[-1].display(False)
print("Population initiale créée")
ATT_liste = ATT_liste_build(liste_reseaux, T)
print("ATT calculée")

[liste_reseaux, ATT_liste] = Ordonner_reseaux(liste_reseaux, ATT_liste)

print(ATT_liste)

### Evolution des solutions ###

nb_iter = 5
N_pop = len(liste_reseaux)      # Taille de la population
p_M = 0.3        # Proportion de grande mutation
p_m = 0.3        # Proportion de petite mutation de taille e_m autour du meilleur : individu ← individu max +U([−e_m ,e_m ])
p_s = 0.3      # Proportion de sexe : individu ← (individu max + individu)/2

print("Starting Optimization")
print(Optimisation(nb_iter, N_pop, p_M, p_m, p_s, liste_reseaux, T))
