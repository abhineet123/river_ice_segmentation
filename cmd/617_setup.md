<!-- MarkdownTOC -->

- [install](#install_)
- [bugs](#bug_s_)

<!-- /MarkdownTOC -->

<a id="install_"></a>
# install
sudo apt install ffmpeg
sudo apt-get install python3.10-tk
pip install imutils scikit-video einops tensorflow
pip install numpy==1.23.5

<a id="bug_s_"></a>
# bugs
`skvideo attributeerror: module 'numpy' has no attribute 'int'.`
pip install numpy==1.23.5

`ModuleNotFoundError: No module named 'keras.src.engine'`
https://stackoverflow.com/a/77649197/10101014
pip install --upgrade tensorflow==2.13

