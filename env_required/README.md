# Some extra specific install instructions on top of pip install -r requirements.txt

#### specific version of torch
pip install torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install inltk
#### word forms
inside the word_forms-master directory:
python setup.py install
