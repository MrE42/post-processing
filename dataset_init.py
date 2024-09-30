from roboflow import Roboflow

rf = Roboflow(api_key="h1PJXvz9OFS8CQcBVFfA")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(12)
dataset = version.download("yolov8")
