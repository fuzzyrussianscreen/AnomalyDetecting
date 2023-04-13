import tkinter.filedialog as fd
from tkinter import *

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from FormVersion import *


class App(Tk):

	def __init__(self):
		super().__init__()
		# tk = Tk()
		self.title("Поиск аномалий гибридными методами")
		self.geometry('1800x1000')
		self.df = [[]]
		self.dfAnomaly = [[]]
		self.dfAnomalyOWL = [[]]

		self.fig, self.axex = plt.subplots(nrows=3, ncols=1, )

		# self.rowconfigure(3, minsize=800, weight=1)
		# self.columnconfigure(3, minsize=800, weight=1)

		fToolBar = Frame(self)
		fSelection = Frame(fToolBar)

		self.labelData = Label(fSelection, font="Courier 14",
							   text="C:/Users/Dimon/PycharmProjects/AnomalyDetecting/Sources/facies_data/NEWBY.csv")
		self.labelOWL = Label(fSelection, font="Courier 14",
							  text="C:/Users/Dimon/PycharmProjects/AnomalyDetecting/Sources/ontology/ontology_empty.owl")

		self.df = startSearch(self.labelData["text"])
		printDF(self.axex, self.df)
		self.drawChart()

		btnSearch = Button(fSelection, font="Courier 14", text="Поиск аномалий", command=self.neuronSearch)
		# self.neuronSearch()
		btnFileData = Button(fSelection, font="Courier 14", text="Выбрать файл данных", command=self.chooseFileData)
		# btnData = Button(fSelection, text="Выбрать данные", command=self.chooseFileData)

		btnFileOWL = Button(fSelection, font="Courier 14", text="Выбрать файл онтологии", command=self.chooseFileOWL)
		btnOWLSearch = Button(fSelection, font="Courier 14", text="Поиск аномалий (онтология)", command=self.OWLSearch)

		self.btnSWRLLoad = Button(fSelection, font="Courier 14", text="Просмотр SWRL", command=self.loadSWRLRules)
		self.btnSWRLSave = Button(fSelection, font="Courier 14", text="Сохранение SWRL", command=self.saveSWRLRules)

		self.listboxSWRL = Listbox()

		fToolBar.grid(row=0, column=0, sticky="ns")
		fSelection.grid(row=1, column=0, sticky="ns")

		btnSearch.grid(row=0, column=3, sticky="ew", padx=5)
		btnOWLSearch.grid(row=1, column=3, sticky="ew", padx=5)

		self.btnSWRLLoad.grid(row=1, column=4, sticky="ew", padx=5)
		# self.btnSWRLSave.grid(row=1, column=4, sticky="ew", padx=5)

		self.labelData.grid(row=0, column=0, sticky="ew", padx=5)
		btnFileData.grid(row=0, column=1, sticky="ew", padx=5)

		self.labelOWL.grid(row=1, column=0, sticky="ew", padx=5)
		btnFileOWL.grid(row=1, column=1, sticky="ew", padx=5)

	def neuronSearch(self):

		self.fig, self.axex = plt.subplots(nrows=3, ncols=1)
		self.dfAnomaly = SearchAnomaly(self.df, self.df[["GR"]])
		printDFNeuron(self.axex, self.df, self.dfAnomaly)
		self.drawChart()

	def OWLSearch(self):
		self.fig, self.axex = plt.subplots(nrows=3, ncols=1)
		self.dfAnomalyOWL = UsingOntology(self.dfAnomaly, self.labelOWL["text"])
		printDFOntology(self.axex, self.df, self.dfAnomaly, self.dfAnomalyOWL)
		self.drawChart()

	def chooseFileData(self):
		filetypes = [("Текстовый файл", "*.txt *.csv"), ("Все файлы", "*.*")]
		filename = fd.askopenfilename(title="Открыть файл",
									  initialdir="C:\\Users\\Dimon\\PycharmProjects\\AnomalyDetecting\\Sources\\facies_data",
									  filetypes=filetypes)
		if filename:
			self.fig, self.axex = plt.subplots(nrows=3, ncols=1)
			self.labelData["text"] = filename
			self.df = startSearch(self.labelData["text"])
			printDF(self.axex, self.df)
			self.drawChart()

	def chooseFileOWL(self):
		filetypes = [("Онтология", "*.owl"), ("Все файлы", "*.*")]
		filename = fd.askopenfilename(title="Открыть файл",
									  initialdir="C:\\Users\\Dimon\\PycharmProjects\\AnomalyDetecting\\Sources\\ontology",
									  filetypes=filetypes)
		if filename:
			self.labelOWL["text"] = filename

	def drawChart(self):

		# self.fig, self.axex = plt.subplots(nrows=3, ncols=1)
		self.fig.subplots_adjust(left=0.040, bottom=0.05, right=0.5, top=0.95)
		self.fig.set_size_inches(10, 9)
		canvas = FigureCanvasTkAgg(self.fig, self)
		canvas.draw()
		canvas.get_tk_widget().grid(row=3, column=0, sticky="ew", padx=5)

		sb = Scrollbar(self, orient=VERTICAL)
		sb.grid(row=3, column=1, sticky=NS)
		canvas.get_tk_widget().config(yscrollcommand=sb.set)
		sb.config(command=canvas.get_tk_widget().yview)

	def loadSWRLRules(self):

		listSWRL = LoadSWRL(self.labelOWL["text"])
		for SWRL in listSWRL:
			self.listboxSWRL.insert(END, str(SWRL))
		self.listboxSWRL.grid(row=3, column=4, sticky="w", padx=5)

		self.btnSWRLLoad.grid_remove()
		self.btnSWRLSave.grid(row=1, column=4, sticky="ew", padx=5)

	def saveSWRLRules(self):

		self.listboxSWRL.grid_remove()

		self.btnSWRLLoad.grid(row=1, column=4, sticky="ew", padx=5)
		self.btnSWRLSave.grid_remove()


if __name__ == "__main__":
	app2 = App()
	app2.mainloop()
