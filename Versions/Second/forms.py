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

		fSelection = Frame(self)
		self.fView = Frame(fSelection)
		self.fRules = Frame(self.fView)

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


		fSelection.grid(row=0, column=0, sticky="nw")
		self.fView.grid(row=4, column=0, sticky="nw", columnspan=2)
		self.fView.grid(row=4, column=0, sticky="nw", columnspan=3)

		self.labelData.grid(row=0, column=0, sticky="nw", padx=5)
		btnFileData.grid(row=0, column=1, sticky="nwe", padx=5)
		btnSearch.grid(row=0, column=2, sticky="nwe", padx=5)

		self.labelOWL.grid(row=1, column=0, sticky="nw", padx=5)
		btnFileOWL.grid(row=1, column=1, sticky="nwe", padx=5)
		btnOWLSearch.grid(row=1, column=2, sticky="nwe", padx=5)


		self.btnSWRLLoad.grid(row=2, column=1, sticky="nw", padx=5)
		# self.btnSWRLSave.grid(row=1, column=4, sticky="ew", padx=5)

		self.list_scrollbar = Scrollbar(self.fView, orient=HORIZONTAL)
		self.listboxSWRL = Listbox(self.fView, width=100, height=45, yscrollcommand=self.list_scrollbar.set)
		self.list_scrollbar.config(command=self.listboxSWRL.yview)


		#self.listboxSWRL.config(wrap=WORD, wraplength=100)

		self.add_button = Button(self.fView, font="Courier 14", text="Добавить", command=self.add_item)
		self.remove_button = Button(self.fView, font="Courier 14", text="Удалить", command=self.remove_item)
		self.entry = Entry(self.fView)



	def add_item(self):
		item = self.entry.get()
		if item.strip():
			self.listboxSWRL.insert(END, item)
			self.entry.delete(0, END)

	def remove_item(self):
		selected_items = self.listboxSWRL.curselection()
		if selected_items:
			for i in reversed(selected_items):
				self.listboxSWRL.delete(i)


	def loadSWRLRules(self):

		listSWRL = LoadSWRL(self.labelOWL["text"])
		for SWRL in listSWRL:
			self.listboxSWRL.insert(END, str(SWRL))
		self.listboxSWRL.grid(row=2, column=2, sticky="nwe", padx=5, columnspan=2)
		self.list_scrollbar.grid(row=3, column=2, sticky="nwe", padx=5, columnspan=2)

		self.entry.grid(row=0, column=2, sticky="nwe", padx=5, columnspan=2)
		self.add_button.grid(row=1, column=2, sticky="nwe", padx=5)
		self.remove_button.grid(row=1, column=3, sticky="nwe", padx=5)

		self.btnSWRLLoad.grid_remove()
		self.btnSWRLSave.grid(row=3, column=1, sticky="nwe", padx=5)

	def saveSWRLRules(self):

		self.listboxSWRL.grid_remove()
		self.list_scrollbar.grid_remove()
		self.entry.grid_remove()
		self.add_button.grid_remove()
		self.remove_button.grid_remove()

		self.btnSWRLLoad.grid(row=3, column=1, sticky="nw", padx=5)
		self.btnSWRLSave.grid_remove()

		new_rules = self.listboxSWRL.get(0, END)

		listSWRL = SaveSWRL(self.labelOWL["text"], new_rules)

		self.listboxSWRL.delete(0, END)


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
		self.fig.subplots_adjust(left=0.040, bottom=0.05, right=0.95, top=0.95)
		self.fig.set_size_inches(10, 8)
		canvas = FigureCanvasTkAgg(self.fig, self.fView)
		canvas.draw()
		canvas.get_tk_widget().grid(row=0, column=0, sticky="nw", padx=5, rowspan=3)

		sb = Scrollbar(self.fView, orient=VERTICAL)
		sb.grid(row=0, column=1, sticky="ns", rowspan=3)
		canvas.get_tk_widget().config(yscrollcommand=sb.set)
		sb.config(command=canvas.get_tk_widget().yview)




if __name__ == "__main__":
	app2 = App()
	app2.mainloop()
