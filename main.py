from flask import Flask, redirect, url_for, request,render_template
import os
import ast
import functions
from werkzeug import secure_filename

app = Flask(__name__)
classifier = None

@app.route("/",methods=["GET","POST"]) #Default 
def index():
    names =   []
    for file in os.listdir("classifiers"):
        names.append(os.fsdecode(file))
    print(names)
    return render_template('home.html' ,names=names)

@app.route("/extract",methods=["POST"])
def loadDataSet():
    if request.method == 'POST':
        #f = request.files['folder']
        f = "dataset"
        functions.save_features(f)
        dataset = functions.loadData("datasets/dataset.csv")
        vals = functions.crossValidation(dataset)
       	return render_template('precision.html',nr = int(vals[0][0]*100) , reg = int(vals[2][0]*100), tr = int(vals[1][0]*100), data = vals )
    else:
        return "err"


@app.route("/detail", methods=["POST"])
def detailModel():
    print("1")
    if request.method == 'POST':
        print("2")
        nom = request.form["nom"]
        data = request.form["data"]
        data =ast.literal_eval(data)
        print(data[0])
        #return "V"
        return render_template('details.html',kap = int(data[0]*100) , f1 = int(data[1]*100), pre = int(data[2]*100),  rec = int(data[3]*100), nom = nom)

@app.route("/save",methods=["POST"])
def savem():
    nom = str(request.form["n"])
    model = str(request.form["nom"])

    dataset = functions.loadData("datasets/dataset.csv")
    if(model=="log"):
        clasif  = functions.trainingLogReg(dataset)
        functions.savemodel(clasif,nom)
    elif (model == "nr"):
        clasif  = functions.trainingNeuralNetwork(dataset)
        functions.savemodel(clasif,nom)
    else:
        clasif  = functions.trainingDecTrees(dataset)
        functions.savemodel(clasif,nom)
    return 'v'
	

@app.route("/upload",methods=["GET"])
def redirect1():
	return render_template('home.html')

@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method == 'POST':
        name = request.form["s"]
        audio = request.files['file']
        audio.save(secure_filename("prediction.wav"))
        return render_template('predict.html',prediction=functions.predict("prediction.wav",str(name)),n=name)
    else:
        name = request.args.get("s")
        return render_template('predict.html',s=name ,prediction="")
        


if __name__ == "__main__":
	app.run(debug=True)