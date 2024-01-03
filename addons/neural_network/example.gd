extends Node

func rl_test() -> void:
	var rlnn = RLNNET.new([1,3,2,1], 0.008, true)
	rlnn.set_input([0.18])
	rlnn.show_result()
	for i in range(500):
		# here I'm calculating reward for what neural network does. In this case the less output, the higher the reward, so neural network will try to minimize the output to maximize the reward
		rlnn.set_reward(1.0 - rlnn.get_output()[0])
		rlnn.update()
	rlnn.show_result()

func basic_test() -> void:
	var test = NNET.new([1,1], 0.1, true)
	test.set_desired_output([0.5])
	test.set_input([0.1])
	test.run()
	test.show_result()
	test.train(500)
	test.show_result()
	


func xor_test() -> void:
	var xor = NNET.new([2,5,5,1],0.1,true)
	
	xor.set_input([1,0])
	xor.set_desired_output([1.0])
	xor.run()
	xor.show_result()
	xor.train(500)
	xor.show_result()
	
	xor.set_input([0,1])
	xor.set_desired_output([1.0])
	xor.run()
	xor.show_result()
	xor.train(500)
	xor.show_result()
	
	xor.set_input([0,0])
	xor.set_desired_output([0.0])
	xor.run()
	xor.show_result()
	xor.train(500)
	xor.show_result()
	
	xor.set_input([1,1])
	xor.set_desired_output([0.0])
	xor.run()
	xor.show_result()
	xor.train(500)
	xor.show_result()


func _ready() -> void:
	print("Here, the neural network tries to make the output equal to 0.5. The training method used is back propagation.")
	basic_test()
	print("Here, the neural network tries to make the output equal to 0.0. The training method used is reinforcement learning.")
	rl_test()
	
