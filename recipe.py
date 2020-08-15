
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox) 
from matplotlib.patches import Rectangle

import os
import re
import numpy as np
import operator
import math
from PIL import Image, ImageFont
import copy
import yaml
import sys
import argparse
sys.path.append("./images/")


def get_screen_resolution():

	output = os.popen("xrandr | grep '*'")
	output = output.read()
	try:
		resolution = re.search(' +([0-9Xx]+) .*$\n', output).group(1)
	except AttributeError:
		resolution = '1920x1080'

	resolution = resolution.split('x')
	return float(resolution[0]), float(resolution[1])

def get_screen_dpi():
	import sys
	from PyQt5.QtWidgets import QApplication
	app = QApplication(sys.argv)
	screen = app.screens()[0]
	dpi = screen.physicalDotsPerInch()
	app.quit()

	return dpi




class SubRegion:

	def __init__(self,height, width, center = True):
		self.height = height
		self.width = width
		self.center = center

	
	@classmethod
	def from_text(cls, string_text, style):
		padding = style['text_padding']
		dpi = 141.5

		font = style['text_font']
		fontsize = style['text_size']
		weight = style['text_weight']

		if weight == 'bold':
			bold_factor = 1.08
		else:
			bold_factor = 1.0
		bold_factor = 1.0
		#breakpoint()
		split_string = string_text.split('\n')
		
		num_lines = len(split_string)
		pixel_between_lines = 6

		if font == 'DejaVu Sans':
			if weight == 'bold':
				img_font = 'DejaVuSans-Bold.ttf'
			else:
				img_font = 'DejaVuSans.ttf'
		else:
			if weight == 'bold':
				img_font = 'DejaVuSerif-Bold.ttf'
			else:
				img_font = 'DejaVuSerif.ttf'

		# height =  1/72*fontsize * dpi * len(split_string) + (num_lines - 1) * pixel_between_lines
		# width = 1/72*fontsize * dpi * max([len(x) for x in split_string]) * 0.45 #0.373
		# # text1 81 x 10 - salted water ----  1.2857142
		# # text2 145 x 43 - 3 T chili garlic sauce
		# # itle 292  27

		font = ImageFont.truetype(img_font, fontsize)
		width_list =[font.getsize(s)[0] for s in split_string]
		height_list = [font.getsize(s)[1] for s in split_string]
		width = max(width_list)
		height = max(height_list) * 1.36

		#breakpoint()
		height = height * num_lines + (num_lines - 1) * pixel_between_lines
		width *= 1.9
		height*= bold_factor
		width *= bold_factor


		return cls(int(height + padding), int(width + padding), center = False)




class ImageObj:
	def __init__(self, file, zoom, ax):

		#breakpoint()
		print(file)
		img = Image.open(file)

		#breakpoint()
		w, h = img.size
		total_size = w * h

		optimal_height = 75
		factor = optimal_height / h
		h1 = int(h * factor)
		w1 = int(w * factor)

		
		img = img.resize((w1,h1))
		self.image_obj =  np.array(list(img.getdata()))

		self.image_obj = self.image_obj.reshape((h1, w1,4))
		#self.image_data = self.image_data.reshape((h1, w1, 4))
		#img = plt.imread(file, format = '.png')
		#height, width = int(img.shape[0]*zoom), int(img.shape[1]*zoom)

		#self.image_obj = img
		#self.image_obj = OffsetImage(img, zoom = zoom)
		#breakpoint()
		#self.image_obj.axes = ax

		self.region = SubRegion(h1, w1)
		#self.region = SubRegion(height, width)




class HeatImage(ImageObj):
	def __init__(self, file, ax):
		super().__init__('images/temp/aa/temp_' + file + '.png', 0.1, ax)

		# zoom = .1
		# heat_img = plt.imread('images/temp/aa/temp_' + file + '.png', format = 'png')
		# height, width = int(heat_img.shape[0]*zoom), int(heat_img.shape[1]*zoom)

		# self.image_obj = OffsetImage(heat_img, zoom = zoom)
		# self.image_obj.axes = ax
		#self.region = SubRegion(height, width)


class Canvas:
	def __init__(self, height, width, style, debug):
		fig = plt.figure()
		self.ax = fig.add_subplot(111)
		self.height = int(height)
		self.width = int(width)
		self.open_region = np.full((self.height, self.width), True, dtype = bool)
		self.vessel_list = {}
		self.debug = debug
		self.style = style

	def draw_rectangle(self, xy, height, width):

		rect = Rectangle(xy, width, height, edgecolor = 'k', fill = False)
		self.ax.add_patch(rect)



	def add_textbox(self, textobj, data = None, rel_object = None, rel_direction = None, align = None,space = None, yoffset = 0, box = False, bold = False):

		if data is not None:
			target_x = data[0]
			target_y = data[1]
			x_disp = textobj.region.width / 2 / self.width
		else:
			
			xpos = rel_object.xpos
			ypos = rel_object.ypos
			if self.debug:
				self.ax.scatter(xpos, ypos, s = 100, c ='b')

			width = rel_object.width / self.width
			height = rel_object.height / self.height

			if rel_direction == 'top':
				target_y = ypos + height/2 + textobj.region.height/2 / self.height
				target_x = xpos
			elif rel_direction == 'bottom':
				target_y = ypos - height/2 - textobj.region.height/2 / self.height - 7/self.height  # - 5 accounts for line and padding around line
				target_x = xpos

			if align == 'left':
				x_disp = width/2
				x_disp = textobj.region.width/2/self.width
				target_x = xpos - width/2 + textobj.region.width/2/self.width
			else:
				x_disp = textobj.region.width / 2 / self.width
			


			target_y += yoffset

		padding = 0
		y_disp = (textobj.region.height/2 + padding*2)/self.height
		#x_disp = width/4
		if self.debug:
			self.ax.scatter(target_x, target_y, s = 100, c ='r')
		x_pos, y_pos = self.find_optimal_placement(textobj.region, target_x, target_y, False, False)

		textobj.region.xpos = x_pos
		textobj.region.ypos = y_pos
		if self.debug:
			self.ax.scatter(x_pos, y_pos, s = 50, c ='g')
		#self.ax.annotate(textobj.text, xy = (x_pos, y_pos),fontsize = textobj.fontsize)
		if box:
			bbox =dict(facecolor='none', edgecolor='black', boxstyle = 'round')
		else:
			bbox = None

		# if bold:
		# 	fontdict = {'weight' :'bold'}
		# else:
		# 	fontdict = {}

		# try:
		# 	fontdict['family'] = textobj.fontfamily
		# except AttributeError:
		# 	pass

		self.ax.text(x_pos - x_disp, y_pos - y_disp, textobj.text, fontsize = textobj.fontsize, bbox = bbox, fontdict = textobj.fontdict)
	


	def add_image(self, imageobj, target_x, target_y, fixed_x = False, fixed_y = False):
		x_pos, y_pos = self.find_optimal_placement(imageobj.padded_region, target_x, target_y, fixed_x, fixed_y)

		imageobj.region.xpos = x_pos
		imageobj.region.ypos = y_pos

		imageobj.padded_region.xpos = x_pos
		imageobj.padded_region.ypos = y_pos
		#breakpoint()
		#abox = AnnotationBbox(imageobj.image_obj, [x_pos, y_pos],frameon = False)
		#breakpoint()
		print(target_x)
		print(x_pos)

		self.ax.imshow(imageobj.image_obj, extent = [x_pos-imageobj.region.width/2/self.width, x_pos + imageobj.region.width/2/self.width,y_pos-imageobj.region.height/2/self.height, y_pos + imageobj.region.height/2/ self.height], aspect = 'auto')
		#self.fig.figimage(imageobj.image_data.astype(np.uint), x_pos*self.width, y_pos*self.height, origin  = 'lower')
		#self.ax.add_artist(abox)
		

	def find_optimal_placement(self, region, target_x, target_y, fixed_x, fixed_y):


		target_x = (target_x) * self.width
		target_y = (target_y * self.height)

		rows, cols = self.get_search_region(region, target_x, target_y, fixed_x, fixed_y)

		half_height = int(region.height / 2)
		half_width = int(region.width / 2)
		valid_regions = {}
		padding = 10

		if fixed_y:
			y_factor = 10
		else:
			y_factor = 2

		if fixed_x:
			x_factor = 10
		else:
			x_factor = 2

		for row in rows:
			for col in cols:
				if np.all(self.open_region[max(0,row - half_height - padding):min(rows[-1],row+half_height+padding), max(0,col - half_width- padding):min(cols[-1],col+half_width+padding)]):
					valid_regions[(row,col)] = math.sqrt((target_x - (col))**x_factor + (target_y - (row))**y_factor)

		try:
			optimal_placement = min(valid_regions.items(), key = operator.itemgetter(1))[0]


		except ValueError:
			print('No valid region found')
			optimal_placement = [target_y, target_x]


		#breakpoint()
		optimal_x = optimal_placement[1] / self.width
		optimal_y = (optimal_placement[0] / self.height)
		self.destroy_region(optimal_placement, region)

		return optimal_x, optimal_y

	
	def get_search_region(self,region, target_x, target_y, fixed_x, fixed_y):
		max_col = self.width - region.width/2
		max_row = self.height - region.height/2

		x_factor = 3
		y_factor = 3
		if fixed_x:
			x_factor = 1
		if fixed_y:
			y_factor = 1

		left = int(max(region.width/2,target_x - region.width*x_factor))
		right = int(min(max_col, target_x + region.width*x_factor))
		top = int(min(max_row, target_y + region.height*y_factor))
		bottom = int(max(region.height/2,target_y - region.height*y_factor))

		return range(bottom, top), range(left, right)

	def destroy_region(self,placement, region, timeline = False):
		x = int(placement[1])
		y = int(placement[0])
		width = int(region.width/2)
		height = int(region.height/2)

		padding = 0
		row_min = int(max(0,y - height - padding))
		row_max = int(min(self.height, y+height + padding))
		if timeline:
			col_min = 0
			col_max = self.width
		else:
			col_min = int(max(0, x - width - padding))
			col_max = int(min(self.width, x + width + padding))
		
		if self.debug:
			self.draw_rectangle((col_min/self.width, row_min/self.height), (height*2 - padding/2)/self.height, (width*2 - padding/2)/self.width)

		self.open_region[row_min:row_max, col_min:col_max] = False
		#self.open_region[col_min:col_max,row_min:row_max] = False

	def draw_zones(self):
		shape = self.open_region.shape
		pixel_map = np.empty([shape[0], shape[1], 4],dtype = np.float32)
		for row in range(shape[0]):
			for col in range(shape[1]):
				if not self.open_region[row][col]:
					pixel_map[shape[0] - row - 1][col] = [1.0, 0.0, 0.0, 0.5]
				else:
					pixel_map[shape[0] - row - 1][col] = [0.0, 0.0, 0.0, 0.0]
		
		self.ax.imshow(pixel_map, interpolation = 'nearest', extent = (0, 1, 0, 1))
		plt.show()
		breakpoint()
	
	def format_plot(self):
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

		# X AXIS -BORDER
		ax.spines['bottom'].set_visible(False)
		# BLUE
		ax.set_xticklabels([])
		# RED
		ax.set_xticks([])
		# RED AND BLUE TOGETHER
		ax.axes.get_xaxis().set_visible(False)

		# Y AXIS -BORDER
		ax.spines['left'].set_visible(False)
		# YELLOW
		ax.set_yticklabels([])
		# GREEN
		ax.set_yticks([])
		# YELLOW AND GREEN TOGHETHER
		ax.axes.get_yaxis().set_visible(False)

		# ax.add_artist(ab)
		ax.set_ylim(0,1)

		ax.set_xlim(-.05,1.05)

		figManager = plt.get_current_fig_manager()
		figManager.window.showMaximized()

	def make_recipe(self, recipe_info, Vessel_list):

		vessel_name = [vessel.name for vessel in Vessel_list]
		vessel_obj = [vessel for vessel in Vessel_list]
		self.vessel_list = dict(zip(vessel_name, vessel_obj))

		# get max step time of all vessesls in order to scale timelines
		max_time = 0

		for vessel in Vessel_list:
			vessel.max_time = vessel.step_times
			# if vessel.step_times[-1] > max_time:
			# 	max_time = vessel.step_times[-1]
			if vessel.steps[-1].end_time > max_time:
				max_time = vessel.steps[-1].end_time



		# sort Vessel list by max step time
		Vessel_list = sorted(Vessel_list, key = lambda vessel: vessel.steps[-1].end_time, reverse = True)
		
		num_vessels = len(Vessel_list)
		y_incr = 1 / (num_vessels + 1)

		left_padding = 0.02
		time_line_padding = 0.01
		
		title_ypos = self.style['Title']['text_height']
		self.add_textbox(recipe_info.title, data = [0.5, title_ypos])
		self.add_textbox(recipe_info.about, data = [0.5, title_ypos - self.style['About']['text_height_diff']])

		for i, vessel, in enumerate(Vessel_list):
			y_val = 1 - y_incr*(i+1) - y_incr/2
			
			time_line_region = SubRegion(4,  self.width*1)

			
			self.destroy_region([y_val*self.height, vessel.steps[-1].end_time / max_time * self.width/2], time_line_region, timeline = True)#(0 + vessel.steps[-1].end_time / max_time * self.width)/2, 
				#y_val*self.height, self.width, 4)
			
			
			
			for step in vessel.steps:

				# Adjust vessel region so that all
				# text fits within x-bounds
				vessel.adjust_width(step)

				begin_time = step.begin_time
				duration = step.duration
				x_val = begin_time/max_time + step.xshift/max_time

				if x_val == 0:
					x_val += left_padding

					#def add_textbox2(self, textobj, data = None, rel_object = None, rel_direction = None, align = None,space = None):


				self.add_image(vessel, x_val, y_val+ time_line_padding, fixed_y = True)

				#self.add_textbox(step, x_val, y_val - time_line_padding, fix_top = True)
				self.add_textbox(step.text, rel_object = vessel.region, rel_direction = 'bottom', bold = True)
				
				if step.duration_text is not None:
					self.add_textbox(step.duration_text, rel_object = step.text.region, rel_direction = 'bottom')
	
				#if step.heat is not None:
				#	self.add_image(step.heat, x_val, y_val + time_line_padding)

				if step.ingredients is not None:
					self.add_textbox(step.ingredients, rel_object = vessel.padded_region, rel_direction = 'top',yoffset = 0.03, box = True)

				if step.transfer is not None:
					self.draw_arrow(vessel, step.transfer, step.transfertime)

			self.ax.hlines(y_val, 0, vessel.region.xpos + (vessel.padded_region.width/2)/self.width)#vessel.steps[-1].end_time / max_time)
		#self.draw_zones()

	def draw_arrow(self, starting_vessel, ending_vessel, time):
		
		start_y = starting_vessel.region.ypos
		end_vessel = self.vessel_list[ending_vessel]
		end_y = end_vessel.region.ypos
		start_x = starting_vessel.region.xpos
		#breakpoint()
		end_x = time / end_vessel.step_times[-1]

		scale = 0.2
		diff_x = (end_x - start_x) * scale
		diff_y = (end_y - start_y) * scale

		if start_y < end_y:
			direction = 1
		else:
			direction = -1
		self.ax.arrow(start_x, start_y + direction*starting_vessel.region.height/2/self.height + .005, diff_x + .035, diff_y)



class RecipeInfo:
	def __init__(self, title, servings, duration, style):
		self.title = TextObj(title, style['Title'])
		servings = 'Servings: ' + str(servings)
		if duration < 100:
			duration_text = str(duration) + ' min.'
		else:
			duration_text = str(duration // 60) + ' hr ' + str(duration % 60) + ' min'

		time = 'Time: ' + str(duration_text) 
		self.about = TextObj('     ' + servings + '    ' + time, style['About'])
		# self.servings = TextObj(, 10)
		# self.time = TextObj('Time: ' + str(total_time), 10)

# class Title:
# 	def __init__(self, title_text):
# 		self.text = title_text
# 		self.fontsize = 20
# 		self.regi
# 		end_x = time / end_vessel.step_times[-1]

# 		scale = 0.45
# 		diff_x = (end_x - start_x) * scale
# 		diff_y = (end_y - start_y) * scale

# 		if start_y < end_y:
# 			direction = 1
# 		else:
# 			direction = -1
# 		self.ax.arrow(start_x, start_y + direction*starting_vessel.region.height/2/self.height, diff_x + .035, diff_y)


class Step:
	def __init__(self, ax, style, ingredients = None, duration = None, heat = None, text = None, transfer = None, transfertime = None,end = False, xshift = None):
		#self.ingredients = ingredients
		#self.ingredients_list = 
		#self.ingredients_region =SubRegion.from_text(self.ingredients_list)

		if ingredients is not None:
			ingredient_text = '\n'.join(ing.text for ing in ingredients)
			self.ingredients = TextObj(ingredient_text, style['Ingredients'])
		else:
			self.ingredients = None
		if duration is not None and heat is not None:
			duration_text = self.format_duration(duration, heat)
			self.duration_text = TextObj(duration_text, style['Step_Duration_Text'])
			self.duration = duration
		elif duration is not None and heat is None:
			self.duration = duration
			self.duration_text = None
		else: 
			self.duration = 0
			self.duration_text = None
		#if heat is not None:
		#	self.heat = HeatImage(heat, ax)

		if text is None:
			text = ''
		self.text = TextObj(self.format_text(text), style['Step_Text'])

		#self.text = self.format_text(text)
		#self.region = SubRegion.from_text(self.text, fontsize = 7, bold = True)
		#self.fontsize = 7
		self.transfer = transfer
		self.transfertime = transfertime
		self.end = end
		self.get_max_width()
		if xshift is not None:
			self.xshift = xshift
		else:
			self.xshift = 0
		#self.text_width = self.get_text_width(self.text)
		print(xshift)
		print('xshift')
	@staticmethod
	def format_text(step_text):
	#	breakpoint()
		new_line_every_x = 4
		split_text = step_text.split(' ')
		for i in range(len(split_text)):
			if (i ) % new_line_every_x == 0 and i != 0:
				split_text[i] = '\n' + split_text[i]

		return ' '.join(split_text)

	def format_duration(self, duration, heat):
		if duration < 100:
			duration_text = str(duration) + ' min.'
		else:
			duration_text = str(duration // 60) + ' hr ' + str(duration % 60) + ' min'

		if heat is not None:
			duration_text += ' on ' + heat

		return duration_text

	def get_max_width(self):
		if self.duration_text is None:
			duration_width = 0
		else:
			duration_width = self.duration_text.region.width

		if self.ingredients is None:
			ingred_width = 0
		else:
			ingred_width = self.ingredients.region.width
		self.max_width = max([self.text.region.width, duration_width, ingred_width])





class TextObj:
	def __init__(self, text, style):
		self.text = text
		self.fontsize = style['text_size']
		self.fontdict = {'weight' : style['text_weight'],
		'family' : style['text_font']}
		self.region = SubRegion.from_text(self.text, style)


class Ingredient2:
	def __init__(self, name, style):
		self.text = self.format_text(name, style)
		self.fontsize = style['text_size']
		self.region = SubRegion.from_text(self.text, style)
		


	@staticmethod
	def format_text(step_text,style):
	#	breakpoint()
		new_line_every_x = style['words_per_line']
		split_text = step_text.split(' ')
		for i in range(len(split_text)):
			if (i ) % new_line_every_x == 0 and i != 0:
				split_text[i] = '\n' + split_text[i]

		return ' '.join(split_text)

class Ingredient:
	def __init__(self, name, amount, prep = None):

		self.text = self.format_str(name, amount, prep)
		self.text = self.format_text(self.text)
		self.fontsize = 9
		self.region = SubRegion.from_text(self.text, fontsize = self.fontsize)
		

	
	def format_str(self, name, amount, prep):
		if prep is None:
			return str(amount) + ' ' + name
		return str(amount) + ' ' + name + ' ' + prep

	@staticmethod
	def format_text(step_text):
	#	breakpoint()
		new_line_every_x = 6
		split_text = step_text.split(' ')
		for i in range(len(split_text)):
			if (i ) % new_line_every_x == 0 and i != 0:
				split_text[i] = '\n' + split_text[i]

		return ' '.join(split_text)


class Vessel(ImageObj):
	def __init__(self, name,image,ax, heat = None):
		self.name = name
		self.image = image

		super().__init__(image, 0.1, ax)

		self.width = self.region.width
	def add_steps(self,Steps):
		self.steps = Steps
		self.get_times()


	def get_times(self):

		self.step_times = [0]
		total_time = 0
		for step in self.steps:
			step.begin_time = total_time
			step.end_time = total_time + step.duration
			total_time = step.end_time
			self.step_times.append(self.step_times[-1] + step.duration)

	def set_timeline(self,xmin,xmax,y):
		self.xmin = xmin
		self.xmax = xmax
		self.y = y


	def adjust_width(self,step):	
		self.padded_region = copy.deepcopy(self.region)
		if step.max_width > self.width:
			self.padded_region.width = step.max_width
		else:
			self.padded_region.width = self.width


def get_recipe_info(input):
	title = input['Title']
	servings = input['Servings']
	duration = input['Duration']

	return RecipeInfo(title, servings, duration, input['Style'])

def extract_value(step_dict,yaml_dict, key):
	try:
		val = yaml_dict[key]
		step_dict[key] = val
	except KeyError:
		pass

	return step_dict

def get_image(name):

	if name == 'Dutch Oven':
		return 'images/dutch_oven.png'
	elif name == 'Cast Iron':
		return 'images/cast_iron.png'
	elif name == 'Large Pot':
		return 'images/large_pot.png'
	elif name == 'Small Pot':
		return 'images/small_pot.png'
	elif name == 'Large Bowl':
		return 'images/large_bowl.png'
	elif name == 'Small Bowl':
		return 'images/small_bowl.png'
	else:
		return 'images/dutch_oven.png'


def create_vessels(input, ax):
	n = len(input['Vessels'])
	vessel_list = []
	style = input['Style']
	keys = ['duration','heat','text','transfer','transfertime', 'xshift']
	for vessel in input['Vessels']:
		steps = []
		n = len(vessel['vessel']['Steps'])

		for i, step in enumerate(vessel['vessel']['Steps']):
			step_dict = {}
			ingreds = []
			try:
				for ingred in step['step']['Ingredients']:
					ingreds.append(Ingredient2(ingred, input['Style']['Ingredients']))
			except KeyError:
				pass

			if len(ingreds) > 0:
				step_dict['ingredients'] = ingreds
			
			

			for key in keys:
				step_dict = extract_value(step_dict, step['step'], key)

			if i == n:
				step_dict['end'] = True

			step_dict['ax'] = ax
			step_dict['style'] = input['Style']

			
			steps.append(Step(**step_dict))
		
		name = vessel['vessel']['name']
		image = get_image(name)

		vessel_obj = Vessel(name, image, ax)
		vessel_obj.add_steps(steps)
		vessel_list.append(vessel_obj)
			

	return vessel_list

if __name__ == '__main__':

	

	parser = argparse.ArgumentParser(description='Build a recipe')
	parser.add_argument('yaml', type=str,
                    help='YAML input file')
	parser.add_argument('--d', action = 'store_true', help = 'Whether to run in debug mode')
	
	args = parser.parse_args()

	with open(args.yaml) as file:
		input = yaml.load(file, Loader = yaml.FullLoader)


	width, height = get_screen_resolution()
	canvas = Canvas(height, width, input['Style'], args.d)
	ax = canvas.ax

	vessel_list = create_vessels(input, ax)


	recipe_info = get_recipe_info(input)
	
	canvas.make_recipe(recipe_info,vessel_list)


	canvas.format_plot()

	plt.show()




	print('Done')