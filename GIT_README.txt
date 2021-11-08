Comment travailler avec Git ?
	
	1. S'assurer que personne n'est déjà en train de travailler sur le projet (sinon ça va être chiant pour récupérer les modifications simultanées des deux personnes)
	
Depuis la ligne de commande Git (Git Bash) :
	
	2. [git pull] pour s'assurer qu'on a bien la dernière version du projet
	
	3. faire les modifications qu'on veut
	
Une fois qu'on a fini de modifier, pour ajouter nos modifs au serveur central :

	4. [git add .] pour ajouter tous les fichiers présents dans le répertoire local (de l'ordinateur sur lequel tu travailles en ce moment) du projet. Nécessaire si des fichiers ont été ajoutés et parfois modifiés. [git status] dans le doute pour afficher les informations.
	
	5. [git commit -m "ton_message"] pour mettre à jour TA version du projet dans TON répertoire sur le serveur git
	
	6. [git push] pour que ta version du projet dans ton répertoire sur le serveur git soit copiée sur la référence centrale du projet


Ne pas hésiter à [git status] pour afficher des infos ^^