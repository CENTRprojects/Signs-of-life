FROM python:3.7.17-bullseye

WORKDIR /signs-of-life

#RUN apt-get update
RUN echo "deb http://archive.debian.org/debian bullseye main contrib non-free" > /etc/apt/sources.list && \
    apt-get -o Acquire::Check-Valid-Until=false update
    
# Install postgreSQL, C++ Build Tools & other requried packages
RUN apt install -y libpq-dev jq wget unzip htop tmux xdg-utils build-essential git vim curl libnspr4 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libgbm1 \
    libxrandr2 \
    libasound2 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* && \
    apt install -f 

# # Install Chrome
# RUN cd /tmp && \
#     wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
#     dpkg -i --force-all google-chrome-stable_current_amd64.deb && \
#     apt -f -y install

# Install actual chrome version (after https://linuxcapable.com/install-google-chrome-on-ubuntu-linux/)
#RUN curl -fSsL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor | tee /usr/share/keyrings/google-chrome.gpg > /dev/null && \
#    echo deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main | tee /etc/apt/sources.list.d/google-chrome.list && \
#    apt update -y && \
#    apt install -y google-chrome-stable

RUN cd /tmp && \
    DOWNLOAD_URL=`curl -s https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json | jq -r ' .versions | [ .[] | select(.version | startswith("146.")) ] | sort_by(.version) | last | .downloads.chrome[] | select(.platform == "linux64") | .url '` && \
    wget "${DOWNLOAD_URL}" -O ./google-chrome-stable_current_amd64.deb && \
    unzip ./google-chrome-stable_current_amd64.deb -d /opt/ && \
    ln -s /opt/chrome-linux64/chrome /usr/local/bin/google-chrome && \
    rm ./google-chrome-stable_current_amd64.deb && \
    # Step 3: Verify installation
    google-chrome --version --no-sandbox
#&& \
#    google-chrome --version

# Install ChromeDriver
#RUN cd /tmp && \
#    VERSION=`google-chrome --version | cut -d' ' -f3` && \
#    wget "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/$VERSION/linux64/chromedriver-linux64.zip" && \
#    unzip chromedriver-linux64.zip && \
#    cp ./chromedriver-linux64/chromedriver /usr/bin/chromedriver

#   DOWNLOAD_URL=$(curl -sS https://github.io | grep -B 10 "$CHROME_VERSION" | grep -oP '"url": "\K[^"]*linux64.zip"') 
RUN cd /tmp && \
#   CHROME_VERSION=`google-chrome --version | cut -d' ' -f3` && \
#   CHROME_MAJOR=$(google-chrome --version | grep -oP '\d+' | head -n1) && \
#   echo "CHROME_VERSION [${CHROME_VERSION}] CHROME_MAJOR [${CHROME_MAJOR}]"; sleep 4 && \ 
#    DOWNLOAD_URL="https://storage.googleapis.com/chrome-for-testing-public/$CHROME_VERSION/linux64/chromedriver-linux64.zip" && \
    DOWNLOAD_URL=`curl -s https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json | jq -r ' .versions | [ .[] | select(.version | startswith("146.")) ] | sort_by(.version) | last | .downloads.chromedriver[] | select(.platform == "linux64") | .url '` && \
    wget "$DOWNLOAD_URL" -O ./chromedriver_linux64.zip && \ 
    unzip -o ././chromedriver_linux64.zip && \
    mv -f chromedriver-linux64/chromedriver /usr/local/bin/chromedriver && \
    chmod +x /usr/local/bin/chromedriver && \ 
    rm -f ./chromedriver_linux64.zip

#RUN cd /tmp &&\
#    LATEST=$(curl -sS https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_STABLE) && \
#    URL="https://storage.googleapis.com/chrome-for-testing-public/$LATEST/linux64/chromedriver-linux64.zip" && \
#    curl -O $URL && \
#    unzip -o chromedriver-linux64.zip && \
#    mv -f chromedriver-linux64/chromedriver /usr/local/bin/chromedriver && \
#    chmod +x /usr/local/bin/chromedriver

# Install pip && dependencies
RUN python -m pip install -U pip

# COPY req_april2020.txt /signs-of-life/
COPY requirements_before_torch.txt /signs-of-life/
RUN pip --timeout 10000 install -r requirements_before_torch.txt

RUN pip --timeout 10000 install torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements_after_torch.txt /signs-of-life/
RUN pip --timeout 10000 install -r requirements_after_torch.txt

COPY requirements_final.txt /signs-of-life/
RUN pip --timeout 10000 install -r requirements_final.txt

# word_forms setup
COPY env_required/word_forms-master/ /signs-of-life/env_required/word_forms-master/

RUN cd env_required/word_forms-master/ && python setup.py install

COPY .docker/setup_tok_models.py /signs-of-life/setup_tok_models.py
RUN python -u /signs-of-life/setup_tok_models.py

COPY app_domains /signs-of-life/app_domains

#ENTRYPOINT ["tail", "-f", "/dev/null"]
CMD ["sh", "entrypoint.sh"]
