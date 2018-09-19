
from saliency.metrics import AUC, SAUC, NSS, CC, KLdiv

CONFIG = {

	'model': {
			'name' : 'DVGG16',
		},

	'train' : {
		'dataset' : 'OSIE',
		'metrics': [AUC, SAUC, NSS, CC, KLdiv],
		# 'metrics': [AUC, KLdiv],

		},

	'eval' : {

		'dataset' : ['KTH', 'CROWD', 'TORONTO', 'CAT2000'],
		# 'dataset' : ['CAT2000'],
		'metrics': [AUC, SAUC, NSS, CC, KLdiv],
		},

	'weights_path' : '/media/ramin/data/duration/weights/',
	# 'eval_path': '/media/ramin/data/duration/eval/weights',
	# 'visualization_path' : '/media/ramin/data/scanpath/visualization-4/'
}


#MODELS = ['DVGG16_CLSTM1-32', 'DVGG16_CLSTM1-64', 'DVGG16_CLSTM2', 'DVGG16_CLSTM4', 'DVGG16_BCLSTM3']
#MODELS = ['DVGG16_CLSTM2', 'DVGG16_CLSTM4', 'DVGG16_BCLSTM3']


MODELS = ['DVGG16_CLSTM1-64']

