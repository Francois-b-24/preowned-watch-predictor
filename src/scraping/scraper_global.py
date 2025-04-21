import time
import datetime
import requests
import pandas as pd
import numpy as np
import os
import requests
from urllib import request
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementClickInterceptedException
import sqlite3


def create_database():
    conn = sqlite3.connect('/Users/f.b/Desktop/Data_Science/preowned-watch-predictor/data/raw/montre.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS montre
                 (id INTEGER PRIMARY KEY, 
                 marque TEXT, 
                 modele TEXT, 
                 mouvement TEXT,
                 matiere_boitier TEXT,
                 matiere_bracelet TEXT, 
                 annee_prod INTEGER,
                 etat TEXT,
                 sexe TEXT,
                 prix TEXT,
                 reserve_de_marche TEXT,
                 diametre TEXT,
                 etencheite TEXT, 
                 matiere_lunette TEXT,
                 matiere_verre TEXT,
                 boucle TEXT, 
                 matiere_boucle TEXT,
                 rouage TEXT,
                 ville TEXT,
                 fonctions TEXT,
                 Date_recup DATE
                 )''')
    conn.commit()
    conn.close()

create_database()


def insert_data(marque, modele, mouvement,matiere_boitier, matiere_bracelet, annee_prod,  etat, sexe, prix, reserve_de_marche, diametre, etencheite, matiere_lunette, matiere_verre, boucle, matiere_boucle, rouage, ville, fonctions, Date_recup):
    conn = sqlite3.connect('/Users/f.b/Desktop/Data_Science/preowned-watch-predictor/data/raw/montre.db')
    c = conn.cursor()
    c.execute("INSERT INTO montre (marque, modele, mouvement,matiere_boitier, matiere_bracelet, annee_prod,  etat, sexe, prix, reserve_de_marche, diametre, etencheite, matiere_lunette, matiere_verre, boucle, matiere_boucle, rouage, ville, fonctions, Date_recup) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", 
              (marque, modele, mouvement,matiere_boitier, matiere_bracelet, annee_prod,  etat, sexe, prix, reserve_de_marche, diametre, etencheite, matiere_lunette, matiere_verre, boucle, matiere_boucle, rouage, ville, fonctions, Date_recup))
    conn.commit()
    conn.close()
    
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
import time
import datetime

def recuperation_donnees(lien, nb_page=0): 
    
    """ 
    Fonction pour récupérer les données de montres sur Chrono24.
    """
    
    # Définir le nombre de pages à parcourir 
    try:
        nb_page = int(input('Combien de page(s) souhaites-tu parcourir : '))
        print(f'Je souhaite parcourir : {nb_page} page(s)')
    except ValueError:
        print("Veuillez entrer un nombre valide.")
        return
    
    # Accéder à la page principale 
    driver = webdriver.Firefox()
    driver.get(lien)
    
    time.sleep(5)

    # Cliquer sur le cookie si présent  
    try:
        cookie = driver.find_element(By.CLASS_NAME, 'js-modal-content')
        cookie.find_element(By.CLASS_NAME, 'btn').click()
    except NoSuchElementException:
        pass
    
    time.sleep(5)
    
    # Cliquer sur le menu déroulant des différentes catégories 
    driver.find_elements(By.CLASS_NAME, 'js-carousel-cell')[0].click()

    time.sleep(5)
    
    # Cliquer sur la catégorie de montre à récupérer (Hommes/Femmes)
    categories = driver.find_element(By.CLASS_NAME, 'col-sm-10')
    categories.find_elements(By.TAG_NAME, 'a')[0].click()
    
    # Récupérer toutes les annonces présentes sur la page 
    try:
        page_globale_montre = driver.find_element(By.ID, 'wt-watches')
        liste_montres = page_globale_montre.find_elements(By.CLASS_NAME, 'js-article-item-container')
    except NoSuchElementException:
        print("Erreur : Impossible de trouver les annonces.")
        return

    longueur = len(liste_montres)
    print(f'La longueur de la première liste à parcourir est : {longueur}')
    
    time.sleep(5)
    
    c = 0
    page = 0
    
    # Parcourir les annonces
    while page < nb_page:
        for elem in range(longueur):
            liste_montres[elem].click()
            c += 1 
            time.sleep(5)
            
            try:
                cookie_2 = driver.find_element(By.CLASS_NAME, 'js-modal-content')
                cookie_2.find_element(By.CLASS_NAME, 'btn-secondary').click()
            except NoSuchElementException:
                pass
            
            # Récupérer les données de l'annonce
            try:
                table = driver.find_element(By.TAG_NAME,'table')
                table_fonction = table.find_elements(By.TAG_NAME,'tbody')
            except NoSuchElementException:
                print("Erreur : Impossible de trouver la table de données.")
                driver.back()
                continue
            
            # Extraction des informations
            try:
                fonctions = table_fonction[4].text
            except IndexError:
                fonctions = ""   
                  
            table_caracteristques = table.text.split('\n')
            caracteristiques_decoupage = [elem.split() for elem in table_caracteristques]
            
            # Extraction des caractéristiques
            def extraire_valeur(mot_cle, index):
                return str(next((valeurs[index:] for valeurs in caracteristiques_decoupage if mot_cle in valeurs), ""))

            marque = extraire_valeur('Marque', 1)
            modele = extraire_valeur('Modèle', 1)
            mouvement = extraire_valeur('Mouvement', 2)
            matiere_boitier = extraire_valeur('Boîtier', 1)
            matiere_bracelet = extraire_valeur('bracelet', -1)
            annee_prod = extraire_valeur('fabrication', 3)
            etat = extraire_valeur('État', 1)
            sexe = extraire_valeur('Sexe', -1)
            prix = extraire_valeur('Prix', 1)
            reserve_de_marche = extraire_valeur('Réserve', 2)
            diametre = extraire_valeur('Diamètre', 1)
            etencheite = extraire_valeur('Étanche', 1)
            matiere_lunette = extraire_valeur('lunette', -1)
            matiere_verre = extraire_valeur('Verre', 2)
            boucle = extraire_valeur('Boucle', 1)
            matiere_boucle = extraire_valeur('Matériau', 4)
            rouage = extraire_valeur('Calibre/Rouages', -1)
            ville = extraire_valeur('Emplacement', 1)
            Date_recup = datetime.date.today()
            
            # Insérer les données dans la base de données
            insert_data(marque, modele, mouvement, matiere_boitier, matiere_bracelet, annee_prod, etat, sexe, prix, reserve_de_marche, diametre, etencheite, matiere_lunette, matiere_verre, boucle, matiere_boucle, rouage, ville, fonctions, Date_recup)

            driver.back()
            
            # Mettre à jour la liste des montres après retour à la page principale
            try:
                page_globale_montre = driver.find_element(By.ID, 'wt-watches')
                liste_montres = page_globale_montre.find_elements(By.CLASS_NAME, 'js-article-item-container')
            except NoSuchElementException:
                continue
        
        if c < longueur:
            continue
        elif c == longueur:
            time.sleep(5)
            driver.find_element(By.PARTIAL_LINK_TEXT, "Continuer").click() 
            current_url = driver.current_url
            driver.get(current_url)
            time.sleep(5)
              
            c = 0
            try:
                page_globale_montre = driver.find_element(By.ID, 'wt-watches')
                liste_montres = page_globale_montre.find_elements(By.CLASS_NAME, 'js-article-item-container')
                longueur = len(liste_montres)
                print('La longueur de la liste est :', longueur)
            except NoSuchElementException:
                continue
            
            page += 1
            print('Nous sommes à la page :', page)
            
            if page == nb_page:
                break
            
    driver.quit()

if __name__ == "__main__":
    create_database()
    lien = 'https://www.chrono24.fr/'
    recuperation_donnees(lien)
