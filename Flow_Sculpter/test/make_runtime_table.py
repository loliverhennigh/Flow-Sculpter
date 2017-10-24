

with open("./figs/runtime_table.txt", "w") as table_file:

  table_file.write("Batch Size & 1 & 2 & 4 & 8 & 16 \\\ \hline \n")

  with open("./figs/residual_network_shape_512x512_batch_size_1.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("Flow Net $512^2$ & %.3f sec " % time)
  with open("./figs/residual_network_shape_512x512_batch_size_2.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec " % time)
  with open("./figs/residual_network_shape_512x512_batch_size_4.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec " % time)
  with open("./figs/residual_network_shape_512x512_batch_size_8.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec " % time)
  with open("./figs/residual_network_shape_512x512_batch_size_16.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec \\\ \n" % time)

  with open("./figs/boundary_network_shape_512x512_batch_size_1.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("Param Net $512^2$ & %.3f sec " % time)
  with open("./figs/boundary_network_shape_512x512_batch_size_2.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec " % time)
  with open("./figs/boundary_network_shape_512x512_batch_size_4.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec " % time)
  with open("./figs/boundary_network_shape_512x512_batch_size_8.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec " % time)
  with open("./figs/boundary_network_shape_512x512_batch_size_16.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec \\\ \n" % time)

  with open("./figs/learn_step_shape_512x512_batch_size_1.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("Learn Step $512^2$ & %.3f sec " % time)
  with open("./figs/learn_step_shape_512x512_batch_size_2.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec " % time)
  with open("./figs/learn_step_shape_512x512_batch_size_4.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec " % time)
  with open("./figs/learn_step_shape_512x512_batch_size_8.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec " % time)
  with open("./figs/learn_step_shape_512x512_batch_size_16.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec \\\ \n" % time)
  #table_file.write("& Nan \\\ \n")

  with open("./figs/residual_network_shape_144x144x144_batch_size_1.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("Flow Net $144^3$ & %.3f sec " % time)
  with open("./figs/residual_network_shape_144x144x144_batch_size_2.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec " % time)
  with open("./figs/residual_network_shape_144x144x144_batch_size_4.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec " % time)
  with open("./figs/residual_network_shape_144x144x144_batch_size_8.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec " % time)
  #with open("./figs/residual_network_shape_144x144x144_batch_size_16.txt", "r") as f:
  #  time = float(f.readlines()[-1])
  #  table_file.write("& %.3f sec \\\ \n" % time)
  table_file.write("& Nan \\\ \n")

  with open("./figs/boundary_network_shape_144x144x144_batch_size_1.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("Param Net $144^3$ & %.3f sec " % time)
  with open("./figs/boundary_network_shape_144x144x144_batch_size_2.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec " % time)
  with open("./figs/boundary_network_shape_144x144x144_batch_size_4.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec " % time)
  with open("./figs/boundary_network_shape_144x144x144_batch_size_8.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec " % time)
  with open("./figs/boundary_network_shape_144x144x144_batch_size_16.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec \\\ \n" % time)

  with open("./figs/learn_step_shape_144x144x144_batch_size_1.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("Learn Step $144^3$ & %.3f sec " % time)
  with open("./figs/learn_step_shape_144x144x144_batch_size_2.txt", "r") as f:
    time = float(f.readlines()[-1])
    table_file.write("& %.3f sec " % time)
  table_file.write("& Nan ")
  table_file.write("& Nan ")
  table_file.write("& Nan \\\ \n")



