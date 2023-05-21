import csv
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def toCsv(filename):
    f = open(filename, 'rt')
    content = f.readlines()
    data = False
    header = ""
    newContent = []

    for line in content:
        if not data:
            if "@attribute" in line.lower():
                attri = line.lower().split()
                columnName = attri[attri.index("@attribute") + 1]
                header = header + columnName + ","
            elif "@data" in line.lower():
                data = True
                # Remove coma
                header = header[:-1]
                header += '\n'
                newContent.append(header)
        else:
            if "'" in line:
                line = line.replace("'", "")
            newContent.append(line)
    return newContent


def readCsv(content):
    rows = csv.reader(content)
    headers = next(rows, None)
    column = {}
    for h in headers:
        column[h] = []
    outrow = []
    index = 0
    for row in rows:
        if row != []:
            outrow.append({})
            for h, value in zip(headers, row):
                column[h].append(value.strip())
                outrow[index][h] = value.strip()
            index += 1
    return (headers, column, outrow)


class Classifier:
    def __init__(self, train_name="", test_name=""):
        self.train_name = train_name
        self.test_name = test_name
        self.attr_values = {}
        # Read test data

    def load_traning_set_file(self):
        (self.header, self.column, self.row) = readCsv(toCsv(self.train_name) )
        self.classes = set(self.column[self.header[-1]])
        self.trainingprobs_dic = {}
        for attr in self.header[0:-1]:
            self.attr_values[attr] = set(self.column[attr])

    def load_test_set_file(self):
        if (self.test_name != ""):
            (_, _, self.test_row) = readCsv(toCsv(self.test_name))

    def classesProbability(self):
        classColumn = self.column[self.header[-1]]
        length = len(classColumn)
        self.prob = {x: classColumn.count(x) / length for x in set(classColumn)}
        return self.prob

    def confusion(self):
        attrs = self.header[0:-1]
        actual = []
        predicted=[]
        for row in self.test_row:
            predicted.append(self.classify(row))
            actual.append(row[self.header[-1]])
        results = confusion_matrix(actual, predicted)
        print('Confusion Matrix :')
        print(results)
        print('Accuracy Score :', accuracy_score(actual, predicted))
        print('Report : ')
        print(classification_report(actual, predicted))

    def classifyTest(self):
        # Run through every instance in test
        for row in self.test_row:
            if row != {}:
                row[self.header[-1]] = self.classify(row)
                print('Test ' + str(self.test_row.index(row) + 1) + ': ' + str(row[self.header[-1]]), end='\n\n')

    #returns conditional probabilities done on training set
    def training(self):
        self.prob = self.classesProbability()

        for clas in self.classes:
            classCount = self.column[self.header[-1]].count(clas)
            attrs = self.header[0:-1]
            for attr in attrs:
               for instance in set(self.column[attr]):
                attrCount = 0
                for r in self.row:
                    attrCount += (r[attr] == instance and r[self.header[-1]] == clas)
                valueCount = len(set(self.column[attr]))
                val = float(attrCount + 1) / (classCount + valueCount);
                self.trainingprobs_dic[clas, attr, instance] = val
        return self.trainingprobs_dic

    def classify(self, instance):
        prob = {}
        attrs = self.header[0:-1]
        for c in self.classes:
            conditions = self.prob[c];
            for attr in attrs:
                conditions *= self.trainingprobs_dic[c, attr, instance[attr]]
            prob[c] = conditions
        clas = max(prob, key=prob.get)
        return clas


    #online training,
    # prob = PI p(xi | c) * p(c)
    def probInstanceInClass(self, instance, clas):
        ans = self.prob[clas]
        classCount = self.column[self.header[-1]].count(clas)
        attrs = self.header[0:-1]
        for attr in attrs:
            attrCount = 0
            for r in self.row:
                attrCount += (r[attr] == instance[attr] and r[self.header[-1]] == clas)
            valueCount = len(set(self.column[attr]))
            val = float(attrCount + 1) / (classCount + valueCount);
            ans *= val;
        return ans
    def save(self,filename = "dic.bin1"):
        with open(filename, 'wb') as handle:
            pickle.dump([self.trainingprobs_dic,self.prob,self.header,self.classes,self.attr_values], handle, protocol=pickle.HIGHEST_PROTOCOL)
    def load(self,filename = "dic.bin1"):
        with open(filename, 'rb') as handle:

            [self.trainingprobs_dic,self.prob,self.header,self.classes,self.attr_values] = pickle.load(handle)
    def printStatistic(self):
        print('Number of instance: ' + str(len(self.row)), end='\n\n')
        print(str(len(self.header)) + ' Attributes: ' + str(self.header), end='\n\n')
        print(str(len(self.classes)) + ' Classes: ' + str(self.classes), end='\n\n')
        # Classes Probability
        print('Classes probability: ' + str(self.prob), end='\n\n')
        return
    def getinputfromuser(self):
        attrs = self.header[0:-1]
        conditions = {}
        for attr in attrs:
            attr_vals = set(self.attr_values[attr]);
            print ("Possible values for attr {}=> {}:".format(attr,attr_vals))
            value = input(" Please Enter attribute value for this {}:".format(attr))
            while value not in attr_vals:
                value= input("Please enter correct attribute value :")
            conditions[attr] = value
        return conditions

def py_nb():
    training_dataset  = input ("please enter training set file name (*.Arff files):")
    c = Classifier(train_name=training_dataset + ".arff")
    c.load_traning_set_file() #load files into memory(or variables)
    #train the model and save the model on disk
    c.training()
    print("#### training has been done! ####")
    trained_model_filename = input("please enter a new name for trained model (*.bin) : ")
    c.save(trained_model_filename +".bin")
    #load the model and then classify
    savedtrainedNb = input("Please enter trained model file name (*.bin files):")
    test_dataset = input("Please enter test set file name (*.Arff files):")
    #load trained model
    c.load(savedtrainedNb +".bin")
    #load dataset
    c.test_name = test_dataset + ".arff";
    c.load_test_set_file()
    #c.classifyTest()
    c.confusion()
    value = c.classify(c.getinputfromuser())
    print("Prediction ({}) is : {}".format(c.header[-1],value))
    decide = input('Would you like to enter more values? (y/n)')
    while decide == 'y':
        value = c.classify(c.getinputfromuser())
        print("Prediction ({}) is : {}".format(c.header[-1], value))
        decide = input('Would you like to enter more values? (y/n)')
        continue
        exit()


if __name__ == "__main__":
    py_nb()
