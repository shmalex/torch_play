
def print_arguments(args):
	print('Arguments:')
	for arg in vars(args):
		print('\t', arg, '=', getattr(args, arg))
