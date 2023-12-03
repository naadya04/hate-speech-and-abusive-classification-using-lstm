import sys
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidgetItem
from Gui import Gui
from Preprocessing import Preprocessing
from Model import Model
from Predict import Predict

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Gui()
        self.ui.setupUi(self)
        self.data = None
        self.data_preprocessed = None
        self.ui.buttonInputData.clicked.connect(self.inputData)
        self.ui.buttonPreprocessing.clicked.connect(self.preProcessing)
        self.ui.buttonKlasifikasiModel.clicked.connect(self.trainModel)
        self.ui.buttonKlasifikasi.clicked.connect(self.makePrediction)     
        
    
    # Input Data
    def inputData(self):
        input = Preprocessing()
        self.data = input.inputDataset()
        file_name = self.data.split("/")[-1]
        self.ui.labelData.setText('Nama file dataset : '+ file_name)
        
    # Melakukan preprocessing 
    def preProcessing(self):
        preprocessing = Preprocessing()
        self.data_preprocessed = preprocessing.prosesData(self.data)
        
        # Menampilkan data_preprocessed pada tablePreprocessing
        self.ui.tablePreprocessing.setRowCount(len(self.data_preprocessed))
        self.ui.tablePreprocessing.setColumnCount(1)
        self.ui.tablePreprocessing.setHorizontalHeaderLabels(['Original Text'])
        self.ui.tablePreprocessing.setColumnWidth(0,441)

        for row_idx, comment_text in enumerate(self.data_preprocessed['original_text']):
            item = QtWidgets.QTableWidgetItem(comment_text)
            self.ui.tablePreprocessing.setItem(row_idx, 0, item)
            self.ui.tablePreprocessing.item(row_idx, 0).setFlags(Qt.ItemIsEnabled)
    
    # Melakukan training dan menampilkan hamming loss
    
    def trainModel(self):
        # Capture the values from the input elements
        dropout = float(self.ui.textbox_dropout.text()) if self.ui.textbox_dropout.text() else 0.2
        hidden_units = int(self.ui.textbox_hidden_units.text()) if self.ui.textbox_hidden_units.text() else 128
        recurrent_dropout = float(self.ui.textbox_recurrent_dropout.text()) if self.ui.textbox_recurrent_dropout.text() else 0.3
        epochs = int(self.ui.textbox_epochs.text()) if self.ui.textbox_epochs.text() else 15
        batch_size = int(self.ui.textbox_batch_size.text()) if self.ui.textbox_batch_size.text() else 64

        # Initialize the Model with hyperparameters from the GUI
        model = Model(dropout=dropout, hidden_units=hidden_units, recurrent_dropout=recurrent_dropout, epochs=epochs, batch_size=batch_size)
        
        # Call the buildModel method to train the model
        eval = model.buildModel(self.data_preprocessed)

        # Call the displayParameters method to display the hyperparameters in the tableParameter
        self.displayParameters()

        # Displaying Hamming Loss (assuming eval contains the hamming loss values, this needs verification)
        self.ui.tableHammingLoss.setRowCount(len(eval))
        self.ui.tableHammingLoss.setColumnCount(2)
        self.ui.tableHammingLoss.setHorizontalHeaderLabels(('Kategori','Nilai'))
        self.ui.tableHammingLoss.setColumnWidth(0,126)
        self.ui.tableHammingLoss.setColumnWidth(1,127)
        
        for row_idx, (eval_kategori, eval_value) in eval.iterrows():
            # Set the parameter name in the first column and make the cell read-only
            self.ui.tableHammingLoss.setItem(row_idx, 0, QTableWidgetItem(str(eval_kategori)))
            self.ui.tableHammingLoss.item(row_idx, 0).setFlags(Qt.ItemIsEnabled)
            # Set the parameter value in the second column and make the cell read-only
            self.ui.tableHammingLoss.setItem(row_idx, 1, QTableWidgetItem(str(eval_value)))
            self.ui.tableHammingLoss.item(row_idx, 1).setFlags(Qt.ItemIsEnabled)
            
    def displayParameters(self):    
        dropout = float(self.ui.textbox_dropout.text())
        hidden_units = int(self.ui.textbox_hidden_units.text())
        recurrent_dropout = float(self.ui.textbox_recurrent_dropout.text())
        epochs = int(self.ui.textbox_epochs.text())
        batch_size = int(self.ui.textbox_batch_size.text())

        parameters  = {
        "dropout": dropout,
        "hidden_units": hidden_units,
        "recurrent_dropout": recurrent_dropout,
        "epochs": epochs,
        "batch_size": batch_size
        }
    
        self.ui.tableParameter.setRowCount(len(parameters))
        self.ui.tableParameter.setColumnCount(2)
        
        self.ui.tableParameter.setHorizontalHeaderLabels(('Parameter','Nilai'))
        
        self.ui.tableParameter.setColumnWidth(0,126)
        self.ui.tableParameter.setColumnWidth(1,127)
    
        for row_idx, (param_name, param_value) in enumerate(parameters.items()):
            # Set the parameter name in the first column and make the cell read-only
            self.ui.tableParameter.setItem(row_idx, 0, QTableWidgetItem(str(param_name)))
            self.ui.tableParameter.item(row_idx, 0).setFlags(Qt.ItemIsEnabled)

            # Set the parameter value in the second column and make the cell read-only
            self.ui.tableParameter.setItem(row_idx, 1, QTableWidgetItem(str(param_value)))
            self.ui.tableParameter.item(row_idx, 1).setFlags(Qt.ItemIsEnabled)            

   
    # Menampilkan hasil prediksi
    def makePrediction(self):
        pred = Predict()
        result = pred.predictText(self.ui.inputBox.text())
        result_str = ','.join(result)
        
        print(self.ui.inputBox.text())
        print(result_str)
        
        if result:
            self.ui.outputBox.setText(result_str)        
        else:
            self.ui.outputBox.setText('Other')
        
# Memulai aplikasi      
def create_app():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()  
    window.show()
    sys.exit(app.exec_())
    
create_app()