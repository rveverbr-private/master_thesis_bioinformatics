## Git clone
Github username
Add to organisation


Add ssh keys to Github
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

git clone git@github.com:guilliottslab/crispyKC.git

Set git username and e-mail
git config --global core.editor "nano"
git config --global --edit

## Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
conda create -n crispyKC
conda activate crispyKC
conda install python=3.11

## VScode
```
Host cp0001
    Hostname cp0001.irc.ugent.be
    Port 22345
    User wouters
    IdentityFile ~/.ssh/id_rsa

Host cn2031
    Hostname cn2031
    User wouters
    IdentityFile ~/.ssh/id_rsa
    ProxyJump cp0001
```

Use a workspace
Install python, jupyter extensions

Python, select interpreter -> crispyKC

Try running a notebook

## crispyKC package
Install crispyKC package in developer mode:
`pip install -e .`

## Data
Raw/preprocessed data in data folder
Intermediate output in output folder
These folders are not version controlled

## Code
Code in the code folder

I use jupyter notebooks a lot. These are not version controled, rather jupytext is used to sync them with a .py file which is version controlled
E.g.: `jupytext --sync code/*.ipynb` to sync all notebooks in a folder
If pulled a new version 
