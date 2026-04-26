package genius.core.uncertainty;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;
import java.util.Random;

/**
 * This class is a helper for the fitnessCross_SAGA function in EstimateUtilityLibrary.
 */

public class Helper_SAGA {
	
	private Domain domain;
	private BidRanking bidRank;
	private Random rnd;
	
	public Helper_SAGA(Domain domain, BidRanking bidRank, Random rnd){
		this.domain = domain;
		this.bidRank = bidRank;
		this.rnd = rnd;
	}
	public AbstractUtilitySpace generateRandomUtilitySpace() {
        AdditiveUtilitySpaceFactory additiveUtilitySpaceFactory = new AdditiveUtilitySpaceFactory(domain);
        List<IssueDiscrete> issues = additiveUtilitySpaceFactory.getIssues();
        for (IssueDiscrete i : issues) {
            additiveUtilitySpaceFactory.setWeight(i, rnd.nextDouble());
            for (ValueDiscrete v : i.getValues())
                additiveUtilitySpaceFactory.setUtility(i, v, rnd.nextDouble());
        }

        // Normalize the weights, since we picked them randomly in [0, 1]
        additiveUtilitySpaceFactory.normalizeWeights();

        // The factory is done with setting all parameters, now return the estimated utility space
        return additiveUtilitySpaceFactory.getUtilitySpace();
    }

    // 交叉と突然変異を行います
    public AbstractUtilitySpace crossover(AdditiveUtilitySpace parent1, AdditiveUtilitySpace parent2) {
        double alpha = 0.3; // BLX-alphaの値
        double mutateProb = 0.005;   // 突然変異確率
        double low, high;
        double w1, w2, wChild;

        AdditiveUtilitySpaceFactory additiveUtilitySpaceFactory = new AdditiveUtilitySpaceFactory(domain);
        List<IssueDiscrete> issues = additiveUtilitySpaceFactory.getIssues();
        for (IssueDiscrete i : issues) {
            // 論点の重み
            w1 = parent1.getWeight(i);
            w2 = parent2.getWeight(i);
            low = Math.min(w1, w2) - alpha * Math.abs(w1 - w2);
            high = Math.max(w1, w2) + alpha * Math.abs(w1 - w2);
            wChild = rnd.nextDouble() * (high - low) + low;
            if (wChild < 0.01) wChild = 0.01;
            additiveUtilitySpaceFactory.setWeight(i, wChild);
            //突然変異
            if (rnd.nextDouble() < mutateProb)
                additiveUtilitySpaceFactory.setWeight(i, rnd.nextDouble());

            for (ValueDiscrete v : i.getValues()) {
                // 選択肢の評価値
                w1 = ((EvaluatorDiscrete) parent1.getEvaluator(i)).getDoubleValue(v);
                w2 = ((EvaluatorDiscrete) parent2.getEvaluator(i)).getDoubleValue(v);
                low = Math.min(w1, w2) - alpha * Math.abs(w1 - w2);
                high = Math.max(w1, w2) + alpha * Math.abs(w1 - w2);
                wChild = rnd.nextDouble() * (high - low) + low;
                if (wChild < 0.01) wChild = 0.01;
                additiveUtilitySpaceFactory.setUtility(i, v, wChild);
                //突然変異
                if (rnd.nextDouble() < mutateProb)
                    additiveUtilitySpaceFactory.setUtility(i, v, rnd.nextDouble());
            }

            /* 正規分布で交叉
            if (rnd.nextBoolean())
                wChild = rnd.nextGaussian() * Math.abs(w1 - w2) + Math.min(w1, w2);
            else
                wChild = rnd.nextGaussian() * Math.abs(w1 - w2) + Math.max(w1, w2);
             */
        }

        // Normalize the weights
        additiveUtilitySpaceFactory.normalizeWeights();

        return additiveUtilitySpaceFactory.getUtilitySpace();
    }

    // 適応度関数
    public double fitness(AbstractUtilitySpace individual, boolean print) {
        List<Double> utilList = new ArrayList<>();  // ランキング下位から上位の予測効用値

        for (Bid b : bidRank.getBidOrder()) {
            utilList.add(individual.getUtility(b));
        }

        // {予測効用値:予測順位} の辞書
        TreeMap<Double, Integer> map = new TreeMap<>(Collections.reverseOrder());
        for (double util : utilList) {
            map.put(util, 1);
        }
        int rank = 1;
        for (Map.Entry<Double, Integer> entry : map.entrySet()) {
            rank += entry.setValue(rank);
        }

        List<Integer> rankList = new ArrayList<>();
        for (double util : utilList) {
            rankList.add(map.get(util));
        }
        Collections.reverse(rankList);  // ランキング上位から下位のBidの予測順位(1,2,3,...って並んでるとうれしい)

        int sqSum = 0;
        for (int i = 0; i < rankList.size(); i++) {
            int diff = i + 1 - rankList.get(i);
            sqSum += diff * diff;
        }
        double spearman = 1.0 - 6.0 * (double) sqSum / (Math.pow(rankList.size(), 3) - rankList.size());
        double lowDiff = Math.abs(bidRank.getLowUtility() - individual.getUtility(bidRank.getMinimalBid()));
        double highDiff = Math.abs(bidRank.getHighUtility() - individual.getUtility(bidRank.getMaximalBid()));

        if (print) {
            System.out.println("spearman = " + spearman + ", lowDiff = " + lowDiff + ", highDiff = " + highDiff);
        }

        return spearman * 10 + (1 - lowDiff) + (1 - highDiff);
    }

    // ルーレット選択
    public List<AbstractUtilitySpace> selectByRoulette(List<AbstractUtilitySpace> population, List<Double> fitnessList, int popSize) {
        List<AbstractUtilitySpace> nextGeneration = new ArrayList<>();

        // 適応度が最大の要素を求める
        double maxFit = -1.0;
        int maxIndex = -1;

        double fitSum = 0.0;
        for (int i = 0; i < fitnessList.size(); i++) {
            double fit = fitnessList.get(i);
            if (maxFit < fit) {
                maxFit = fit;
                maxIndex = i;
            }
            fitSum += fit;
        }

        ////System.out.print("average = " + fitSum / population.size() + ", max = " + maxFit + ", ");
        ////fitness(population.get(maxIndex), true);

        nextGeneration.add(population.get(maxIndex));

        for (int i = 0; i < popSize - 1; i++) {
            double randomNum = rnd.nextDouble() * fitSum;
            double count = 0.0;
            for (int n = 0; n < population.size(); n++) {
                count += fitnessList.get(n);
                if (count > randomNum) {
                    nextGeneration.add(population.get(n));
                    break;
                }
            }
        }

        return nextGeneration;
    }
	
}
