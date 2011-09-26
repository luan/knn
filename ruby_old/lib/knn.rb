require 'logger'
require 'parallel'
require 'benchmark'

class Knn
  def initialize(training, testing, k, log=true)
    @training_file = training
    @testing_file = testing
    @k = k.to_i

    @trained = false

    if log
      @log = Logger.new(STDERR)
      @log.datetime_format = "%H:%M "
    end
  end

  def train
    @training = parse_file(@training_file)

    @trained = true
  end

  def run
    train unless @trained

    testing = parse_file(@testing_file)

    i = 0
    testing.each do |sample|
      sample[:distance_vector] = @training.map do |base|
        {
          :distance => euclidean_distance(base[:features], sample[:features]),
          :type => base[:type]
        }
      end.sort_by { |e| e[:distance] }[0, @k]

      types = sample[:distance_vector].map { |e| e[:type] }.uniq
      counter = {}

      types.each do |type|
        counter[type] = sample[:distance_vector].count { |e| e[:type] == type }
      end

      index = counter.max_by { |k, v| v }[0]
      sample[:calculated_type] = @training[index][:type]
    end

    log testing.map {|t| [ t[:calculated_type], t[:type] ]}.to_s
  end

  private

  def euclidean_distance(u, v)
    sum = 0

    u.zip(v).each do |x, y|
      sum += (y - x) ** 2
    end

    Math.sqrt(sum)
  end

  def parse_file(filename)
    log "Parsing file '#{filename}'"

    samples = []
    File.open(filename) do |file|
      head = file.readline.split.map(&:to_i)

      file.each_line do |line|
        line = line.split
        samples << {
          :features => line[0, head[1]].map(&:to_i),
          :type     => line.last.to_i
        }
      end
    end

    samples
  end

  def log(message, type=:info)
    method = caller[0][/`.*'/][1..-2]
    @log.send(type, method) { message } if @log
  end
end

