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
    conn = sqlite3.connect('/Users/f.b/Desktop/Data_Science/watch_market/watch_price_prediction/data/raw/montre.db')
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
                 Date_recup DATE,
                 descriptions TEXT
                 )''')
    conn.commit()
    conn.close()

create_database()


def insert_data(marque, modele, mouvement,matiere_boitier, matiere_bracelet, annee_prod,  etat, sexe, prix, reserve_de_marche, diametre, etencheite, matiere_lunette, matiere_verre, boucle, matiere_boucle, rouage, ville, fonctions, Date_recup, descriptions):
    conn = sqlite3.connect('/Users/f.b/Desktop/Data_Science/watch_market/watch_price_prediction/data/raw/montre.db')
    c = conn.cursor()
    c.execute("INSERT INTO montre (marque, modele, mouvement,matiere_boitier, matiere_bracelet, annee_prod,  etat, sexe, prix, reserve_de_marche, diametre, etencheite, matiere_lunette, matiere_verre, boucle, matiere_boucle, rouage, ville, fonctions, Date_recup, descriptions) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", 
              (marque, modele, mouvement,matiere_boitier, matiere_bracelet, annee_prod,  etat, sexe, prix, reserve_de_marche, diametre, etencheite, matiere_lunette, matiere_verre, boucle, matiere_boucle, rouage, ville, fonctions, Date_recup, descriptions))
    conn.commit()
    conn.close()

def recuperation_donnees(debut, nb_page=0): 
    """ Cette fonction a pour objet : 
    - Se rendre sur la page d'accueil du site Chrono24 
    - Cliquer sur le cookie lorsque ce dernier apparaît
    - Cliquer sur la rubrique de montres à choisir : Hommes/Femmes
    - Se diriger sur la page qui contient toutes les annonces 
    - Récupérer les informations de chaque annonce de manière itérative
    - Stocker ces éléments dans une base de données.
    """
    
    # Définissons le nombre de pages à parcourir : 
    nb_page = int(input('Choix du nombre de pages à parcourir : '))
    print('Je souhaite parcourir :', nb_page, 'page(s)')

    lien = 'https://www.chrono24.fr/watches/montres-femmes--46-'+debut+'.htm'
    driver = webdriver.Firefox()
    driver.get(lien)
    
    time.sleep(5)

    # On clique sur le cookie si disponible  
    try:
        cookie = driver.find_element(By.CLASS_NAME, 'js-modal-content')
        cookie.find_element(By.CLASS_NAME, 'btn').click()
    except NoSuchElementException:
        pass
    
    time.sleep(5)
    
    # On récupère toutes les annonces présentes sur la page 
    page_globale_montre = driver.find_element(By.ID, 'wt-watches')
    liste_montres = page_globale_montre.find_elements(By.CLASS_NAME, 'js-article-item-container')

    longueur = len(liste_montres)
    print('La longueur de la première liste est :', longueur)
    
    time.sleep(5)
    
    c = 0
    page = 0
    
    # On parcourt les éléments un à un
    while True:
        for elem in range(len(liste_montres)):
            try: 
                liste_montres[elem].click()
            except ElementClickInterceptedException:
                continue
            
            c += 1 
            time.sleep(5) 
            
            try:
                cookie_2 = driver.find_element(By.CLASS_NAME, 'js-modal-content')
                cookie_2.find_element(By.CLASS_NAME, 'btn-secondary').click()
            except NoSuchElementException:
                pass
            
            try:
                table = driver.find_element(By.TAG_NAME,'table')
                table_fonction = table.find_elements(By.TAG_NAME,'tbody') 
                
                fonctions = table_fonction[4].text if len(table_fonction) > 4 else ""
                desc = driver.find_elements(By.TAG_NAME,'table')
                descript = desc[1].text
                table_caracteristiques = table.text.split('\n')
            except NoSuchElementException:
               continue
            
            # Récupération des données
            caracteristiques_decoupage = [elem.split() for elem in table_caracteristiques]

            # Extraction des caractéristiques
            def extraire_caracteristique(mot_cle, decoupe):
                return str(next((valeurs[1:] for valeurs in decoupe for elem in valeurs if elem == mot_cle), ""))
            
            
            descriptions = descript
            marque = extraire_caracteristique('Marque', caracteristiques_decoupage)
            modele = extraire_caracteristique('Modèle', caracteristiques_decoupage)
            mouvement = extraire_caracteristique('Mouvement', caracteristiques_decoupage)
            matiere_boitier = extraire_caracteristique('Boîtier', caracteristiques_decoupage)
            matiere_bracelet = extraire_caracteristique('bracelet', caracteristiques_decoupage)
            annee_prod = extraire_caracteristique('fabrication', caracteristiques_decoupage)
            etat = extraire_caracteristique('État', caracteristiques_decoupage)
            sexe = extraire_caracteristique('Sexe', caracteristiques_decoupage)
            prix = extraire_caracteristique('Prix', caracteristiques_decoupage)
            reserve_de_marche = extraire_caracteristique('Réserve', caracteristiques_decoupage)
            diametre = extraire_caracteristique('Diamètre', caracteristiques_decoupage)
            etencheite = extraire_caracteristique('Étanche', caracteristiques_decoupage)
            matiere_lunette = extraire_caracteristique('lunette', caracteristiques_decoupage)
            matiere_verre = extraire_caracteristique('Verre', caracteristiques_decoupage)
            boucle = extraire_caracteristique('Boucle', caracteristiques_decoupage)
            matiere_boucle = extraire_caracteristique('Matériau', caracteristiques_decoupage)
            rouage = extraire_caracteristique('Calibre/Rouages', caracteristiques_decoupage)
            ville = extraire_caracteristique('Emplacement', caracteristiques_decoupage)
            Date_recup = datetime.date.today()
            
            insert_data(marque, 
                        modele, 
                        mouvement, 
                        matiere_boitier, 
                        matiere_bracelet, 
                        annee_prod, 
                        etat, 
                        sexe, 
                        prix, 
                        reserve_de_marche, 
                        diametre, 
                        etencheite, 
                        matiere_lunette, 
                        matiere_verre, 
                        boucle, 
                        matiere_boucle, 
                        rouage, 
                        ville, 
                        fonctions, 
                        Date_recup,
                        descriptions)
            
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
    debut = input('Veuillez entrer le numéro de page à laquelle vous souhaitez débuter (1,2,3, etc.) : ') 
    recuperation_donnees(debut)
