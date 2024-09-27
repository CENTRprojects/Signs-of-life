FROM python:3.7.17-bullseye

WORKDIR /signs-of-life

COPY req_april2020.txt /signs-of-life/
COPY requirements_after_torch.txt /signs-of-life/
COPY requirements_before_torch.txt /signs-of-life/
COPY requirements_final.txt /signs-of-life/



RUN apt-get update

# Install postgreSQL, C++ Build Tools & other requried packages
RUN apt install -y libpq-dev wget htop tmux xdg-utils build-essential git vim && \
    apt install -f 

# # Install Chrome
# RUN cd /tmp && \
#     wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
#     dpkg -i --force-all google-chrome-stable_current_amd64.deb && \
#     apt -f -y install

ARG CHROME_VERSION="128.0.6613.113-1"
RUN wget --no-verbose -O /tmp/chrome.deb https://dl.google.com/linux/chrome/deb/pool/main/g/google-chrome-stable/google-chrome-stable_${CHROME_VERSION}_amd64.deb \
  && apt install -y /tmp/chrome.deb \
  && rm /tmp/chrome.deb



# Install ChromeDriver
RUN cd /tmp && \
    VERSION=`google-chrome --version | cut -d' ' -f3` && \
    wget "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/$VERSION/linux64/chromedriver-linux64.zip" && \
    unzip chromedriver-linux64.zip && \
    cp ./chromedriver-linux64/chromedriver /usr/bin/chromedriver

# Install pip && dependencies
RUN python -m pip install -U pip
RUN pip --timeout 10000 install -r requirements_before_torch.txt
RUN pip --timeout 10000 install torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip --timeout 10000 install -r requirements_after_torch.txt
RUN pip --timeout 10000 install -r requirements_final.txt

# word_forms setup
COPY env_required/word_forms-master/ /signs-of-life/env_required/word_forms-master/

RUN cd env_required/word_forms-master/ && python setup.py install

COPY .docker/setup_tok_models.py /signs-of-life/setup_tok_models.py
RUN python -u /signs-of-life/setup_tok_models.py

#ENTRYPOINT ["tail", "-f", "/dev/null"]
CMD ["sh", "entrypoint.sh"]
