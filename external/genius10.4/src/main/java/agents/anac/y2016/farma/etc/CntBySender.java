package agents.anac.y2016.farma.etc;

import java.util.ArrayList;
import java.util.HashMap;

public class CntBySender {
	private int selectWeight; // どの重みを適用するか

	private ArrayList<Object> opponents;
	private HashMap<Object, Integer> simpleCnt;
	private int simpleSum;
	private HashMap<Object, Double> weightedCnt;
	private double weightedSum;

	public CntBySender(int selectweight) {
		this.selectWeight = selectweight;

		this.opponents = new ArrayList<Object>();
		this.simpleCnt = new HashMap<Object, Integer>();
		this.simpleSum = 0;

		this.weightedCnt = new HashMap<Object, Double>();
		this.weightedSum = 0.0;
	}

	/**
	 * カウント情報の更新
	 * 
	 * @param sender
	 * @param time
	 */
	public void incrementCnt(Object sender, double time) {
		simpleSum += 1;
		if (!simpleCnt.containsKey(sender)) {
			simpleCnt.put(sender, 1);
			opponents.add(sender);
		} else {
			simpleCnt.put(sender, simpleCnt.get(sender) + 1);
		}

		double addCnt = calWeightedIncrement(time);
		weightedSum += addCnt;
		if (!weightedCnt.containsKey(sender)) {
			weightedCnt.put(sender, addCnt);
		} else {
			weightedCnt.put(sender, weightedCnt.get(sender) + addCnt);
		}
	}

	static final int DownLiner = 0;

	public double calWeightedIncrement(double time) {
		double ans = 0.0;
		switch (selectWeight) {
		case DownLiner:
			ans = 1.0 - time;
			break;
		default:
			System.out.println("ERROR: 想定外の重み関数が指定されました。");
		}
		return ans;
	}

	public int getSimpleCnt(Object sender) {
		if (simpleCnt.containsKey(sender)) {
			return simpleCnt.get(sender);
		} else {
			return 0;
		}
	}

	public int getSimpleSum() {
		return simpleSum;
	}

	public double getWeightedCnt(Object sender) {
		if (weightedCnt.containsKey(sender)) {
			return weightedCnt.get(sender);
		} else {
			return 0.0;
		}
	}

	public double getWeightedSum() {
		return weightedSum;
	}

	public boolean isContainOpponents(Object sender) {
		return opponents.contains(sender);
	}

}
