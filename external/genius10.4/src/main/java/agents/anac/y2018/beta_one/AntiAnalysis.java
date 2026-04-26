package agents.anac.y2018.beta_one;

import java.util.List;

import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Queue;

import genius.core.analysis.BidPoint;
import genius.core.analysis.BidSpace;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.misc.Range;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;

public class AntiAnalysis
{
	private AdditiveUtilitySpace utilitySpace;
	private AdditiveUtilitySpace anti;

	private BidSpace antiSpace;
	private List<BidPoint> antiPareto;
	private BidPoint antiKalai;
	private double antiKalaiUtil;

	private double observeDuration;
	private double selfishPoint;
	private double boxSize;

	public AntiAnalysis(AdditiveUtilitySpace utilitySpace, double observeDuration, double selfishRatio, double boxSize)
	{
		this.utilitySpace = utilitySpace;
		this.observeDuration = observeDuration;
		this.boxSize = boxSize;

		createAnti();
		createAntiSpace();

		setSelfishPoint(selfishRatio);
	}

	public void setSelfishPoint(double selfishRatio)
	{
		selfishPoint = lerp(antiKalaiUtil, 1, selfishRatio);
	}

	public Range getBox(double time, boolean betrayed)
	{
		if (time > 1)
		{
			return new Range(selfishPoint, 1);
		}
		else if (time < observeDuration)
		{
			double t = time / observeDuration;

			double lower = lerp(1, selfishPoint, t);
			double upper = clamp(lower + boxSize, lower, 1);

			return new Range(lower, upper);
		}
		else if (!betrayed)
		{
			double t = (time - observeDuration) / (1 - observeDuration);

			double lower = lerp(selfishPoint, antiKalaiUtil - boxSize / 2, t);
			double upper = lerp(selfishPoint + boxSize, antiKalaiUtil + boxSize / 2, t);

			return new Range(lower, upper);
		}
		else
		{
			return getBox(99, true);
		}
	}

	public static double lerp(double a, double b, double t)
	{
		if (t < 0)
		{
			t = 0;
		}
		else if (t > 1)
		{
			t = 1;
		}

		return a + (b - a) * t;
	}

	public static double clamp(double d, double min, double max)
	{
		if (d < min)
		{
			return min;
		}

		if (d > max)
		{
			return max;
		}

		return d;
	}

	private void createAnti()
	{
		anti = (AdditiveUtilitySpace) utilitySpace.copy();

		for (int i = 1; i <= utilitySpace.getNrOfEvaluators(); i++)
		{
			EvaluatorDiscrete evaluator;

			if (utilitySpace.getEvaluator(i) instanceof EvaluatorDiscrete)
			{
				evaluator = (EvaluatorDiscrete) utilitySpace.getEvaluator(i);
			}
			else
			{
				antiKalaiUtil = 0.5;
				return;
			}

			Queue<ValueDataPair> queue = new PriorityQueue<ValueDataPair>(valueDataComparator);

			for (ValueDiscrete value : evaluator.getValues())
			{
				ValueDataPair pair = new ValueDataPair(value, evaluator.getValue(value));
				queue.add(pair);
			}

			ValueDataPair[] pairArray = queueToArray(queue);
			interchangeArray(pairArray);

			EvaluatorDiscrete antiEva = (EvaluatorDiscrete) anti.getEvaluator(i);

			for (ValueDataPair pair : pairArray)
			{
				antiEva.setEvaluation(pair.value, pair.data);
			}
		}
	}

	private class ValueDataPair
	{
		public Value value;
		public int data;

		public ValueDataPair(Value value, int data)
		{
			this.value = value;
			this.data = data;
		}
	}

	private static Comparator<ValueDataPair> valueDataComparator = new Comparator<ValueDataPair>()
	{
		public int compare(ValueDataPair p1, ValueDataPair p2)
		{
			return p1.data - p2.data;
		}
	};

	private ValueDataPair[] queueToArray(Queue<ValueDataPair> queue)
	{
		ValueDataPair[] array = new ValueDataPair[queue.size()];

		for (int i = 0; i < array.length; i++)
		{
			array[i] = queue.poll();
		}

		return array;
	}

	private void interchangeArray(ValueDataPair[] array)
	{
		interchangeArray(array, 0, array.length - 1);
	}

	private void interchangeArray(ValueDataPair[] array, int bot, int top)
	{
		if (top <= bot)
		{
			return;
		}

		ValueDataPair botPair = array[bot];
		ValueDataPair topPair = array[top];

		int botData = botPair.data;
		botPair.data = topPair.data;
		topPair.data = botData;

		array[bot] = botPair;
		array[top] = topPair;

		interchangeArray(array, bot + 1, top - 1);
	}

	private void createAntiSpace()
	{
		try
		{
			antiSpace = new BidSpace(utilitySpace, anti);
			antiPareto = antiSpace.getParetoFrontier();
			antiKalai = antiSpace.getKalaiSmorodinsky();
			antiKalaiUtil = antiKalai.getUtilityA();
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}

	public double getSelfishSlope(double discountFactor)
	{
		double dUtil = (selfishPoint + boxSize / 2) - 1;
		double dt = discountFactor;

		return dUtil / dt;
	}

	public List<BidPoint> getAntiPareto()
	{
		return antiPareto;
	}

	public BidPoint getAntiKalai()
	{
		return antiKalai;
	}

	public double getAntiKalaiUtil()
	{
		return antiKalaiUtil;
	}

	public double getSelfishPoint()
	{
		return selfishPoint;
	}
}
