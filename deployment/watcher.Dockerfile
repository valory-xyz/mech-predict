FROM python:3.10-slim

COPY scripts/mech_watcher.py /root/mech_watcher.py
RUN pip install propel-client==0.0.14

CMD ["python", "/root/mech_watcher.py"]
