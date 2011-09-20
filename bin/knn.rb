#!/usr/bin/env ruby

require 'knn'

training, testing, k, = ARGV

knn = Knn.new(training, testing, k)

knn.run
