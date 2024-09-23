1. env requirement:
```
pip instal trl
```

2. rm old gradio web serve
```python
rm /usr/local/lib/python3.11/dist-packages/fastchat/serve/gradio_web_server.py
```

3. link new gradio web serve
```python
 ln -s web-demo/main-demo/june_gradio_web_server.py /usr/local/lib/python3.11/dist-packages/fastchat/serve/gradio_web_server.py
```

4. run the start serve script
```bash
bash start-demo.sh
```