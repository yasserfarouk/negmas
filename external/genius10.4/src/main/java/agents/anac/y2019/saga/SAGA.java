package agents.anac.y2019.saga;

import java.util.*;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.uncertainty.AdditiveUtilitySpaceFactory;
import genius.core.uncertainty.BidRanking;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.uncertainty.UserModel;


public class SAGA extends AbstractNegotiationParty {
    private Random rnd = new Random();

    private Bid lastOffer;
    private boolean isFirst = true;
    private double firstUtil = 0.95;    // 相手の最初のofferの効用値

//    // テスト用
//    int round = -1;
//    private SortedOutcomeSpace sortedOS;

    @Override
    public void init(NegotiationInfo info) {
        super.init(info);
    }

    @Override
    public Action chooseAction(List<Class<? extends Action>> possibleActions) {
        double time = timeline.getTime();
        double target = getTarget(time);

        ////System.out.println("target = " + target);

        // Check for acceptance if we have received an offer
        if (lastOffer != null) {
            double util = utilitySpace.getUtility(lastOffer);
            if (isAcceptable(time, target, util))
                return new Accept(getPartyId(), lastOffer);
        }

//        // 実験時に見やすいようにスリープ
//        try {
//            Thread.sleep(50);
//        } catch (Exception e) {
//            System.out.println(e);
//        }

        // Otherwise, send out a random offer above the target utility
        return new Offer(getPartyId(), generateRandomBidAboveTarget(target));

//        // テスト用 線形に時間譲歩
//        return new Offer(getPartyId(), sortedOS.getBidNearUtility(1 - timeline.getTime()).getBid());

//        // テスト用 BidRankingのBidを順に提案
//        List<Bid> bidOrd = userModel.getBidRanking().getBidOrder();
//        if (round < bidOrd.size() - 1) round++;
//        return new Offer(getPartyId(), bidOrd.get(round));
    }

    // accept関数
    private boolean isAcceptable(double time, double target, double util) {
        // RV以下の提案はすべてReject
        if (util < utilitySpace.getReservationValue()) return false;

        double timeA = 0.6; // target以下にAccept率を与え始める時刻
        double timeB = 0.997; // すべてのbidにAccept率を与え始める時刻

        if (time <= timeA) {
            double acceptProb = Math.pow((util - target) / (1.0 - target), Math.pow(3, (0.5 - time) * 2));
            return (rnd.nextDouble() < acceptProb);
        } else if (time >= timeB) {
            double acceptProb = Math.pow(util, 2);
            return (rnd.nextDouble() < acceptProb);
        }

        // 時刻が timeA < time < timeB のとき
        double APatT = 0.15 * Math.pow((time - timeA) / (1 - timeA), 2);
        double acceptBase = target - (1 - target) * (time - timeA) / (1 - timeA);

        if (util > target) {
            double acceptProb = APatT + (1.0 - APatT) * Math.pow((util - target) / (1.0 - target), Math.pow(3, (0.5 - time) * 2));
            return (rnd.nextDouble() < acceptProb);
        } else if (util > acceptBase) {
            double acceptProb = APatT * Math.pow((util - acceptBase) / (target - acceptBase), 2);
            return (rnd.nextDouble() < acceptProb);
        }
        return false;
    }

    // 譲歩関数
    private double getTarget(double time) {
        double A = 0.6; // どこまで譲歩するか決めるパラメータ
        double B = 5;   // 譲歩速度(1-time^B)

        double targetMin = firstUtil + A * (1 - firstUtil);
        if (targetMin < utilitySpace.getReservationValue())
            targetMin = utilitySpace.getReservationValue();
        return targetMin + (1.0 - targetMin) * (1.0 - Math.pow(time, B));
    }

    private Bid generateRandomBidAboveTarget(double target) {
        Bid randomBid;
        double util;
        int i = 0;
        int maxLoop = 500;

        // try to find a bid above the target utility
        do {
            randomBid = generateRandomBid();
            util = utilitySpace.getUtility(randomBid);
        }
        while (util < target && i++ < maxLoop);

        try {
            if (i >= maxLoop)
                return utilitySpace.getMaxUtilityBid();
        } catch (Exception e) {
            // Bidが一つもないドメインなんて普通ありえないのでここには来ないはず
            System.out.println("Exception in generateRandomBidAboveTarget:" + e);
        }
        return randomBid;
    }

    @Override
    public void receiveMessage(AgentID sender, Action action) {
        if (action instanceof Offer) {
            lastOffer = ((Offer) action).getBid();

            if (isFirst) {
                firstUtil = utilitySpace.getUtility(lastOffer);
                isFirst = false;
            }
        }
    }

    @Override
    public String getDescription() {
        return "Simple Agent based on Genetic Algorithm";
    }

    // ここをGAでやってみたいと思います
    @Override
    public AbstractUtilitySpace estimateUtilitySpace() {
        int popSize = 500;  // 集団サイズ
        int maxGeneration = 200;   // 打ち切り世代数
        double crossRate = 3.0; // 交叉回数 = popSize * crossRate

        // 初期集団の生成
        List<AbstractUtilitySpace> population = new ArrayList<>();
        for (int i = 0; i < popSize * (1.0 + crossRate); i++) {
            population.add(generateRandomUtilitySpace());
        }

        // maxGeneration世代まで繰り返す
        for (int gen = 0; gen < maxGeneration; gen++) {
            ////System.out.print("gen " + gen + ": ");

            // 適応度関数の計算
            List<Double> fitnessList = new ArrayList<>();
            for (AbstractUtilitySpace ind : population) {
                fitnessList.add(fitness(ind, false));
            }

            // ルーレット選択
            population = selectByRoulette(population, fitnessList, popSize);

            // 交叉と突然変異
            int parentSize = population.size();
            for (int i = 0; i < popSize * crossRate; i++) {
                AbstractUtilitySpace parent1 = population.get(rnd.nextInt(parentSize));
                AbstractUtilitySpace parent2 = population.get(rnd.nextInt(parentSize));
                AbstractUtilitySpace child = crossover((AdditiveUtilitySpace) parent1, (AdditiveUtilitySpace) parent2);
                population.add(child);
            }
        }

        // 適応度関数の計算
        List<Double> fitnessList = new ArrayList<>();
        for (AbstractUtilitySpace ind : population) {
            fitnessList.add(fitness(ind, false));
        }

        // 適応度が最大の要素を求める
        double maxFit = -1.0;
        int maxIndex = -1;
        for (int i = 0; i < population.size(); i++) {
            if (fitnessList.get(i) > maxFit) {
                maxFit = fitnessList.get(i);
                maxIndex = i;
            }
        }

        ////System.out.println("最終結果:\n" + population.get(maxIndex).toString());

//        // テスト用
//        sortedOS = new SortedOutcomeSpace(population.get(maxIndex));

        return population.get(maxIndex);
    }

    private AbstractUtilitySpace generateRandomUtilitySpace() {
        AdditiveUtilitySpaceFactory additiveUtilitySpaceFactory = new AdditiveUtilitySpaceFactory(getDomain());
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
    private AbstractUtilitySpace crossover(AdditiveUtilitySpace parent1, AdditiveUtilitySpace parent2) {
        double alpha = 0.3; // BLX-alphaの値
        double mutateProb = 0.005;   // 突然変異確率
        double low, high;
        double w1, w2, wChild;

        AdditiveUtilitySpaceFactory additiveUtilitySpaceFactory = new AdditiveUtilitySpaceFactory(getDomain());
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
    private double fitness(AbstractUtilitySpace individual, boolean print) {
        BidRanking bidRank = userModel.getBidRanking();
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
    private List<AbstractUtilitySpace> selectByRoulette(List<AbstractUtilitySpace> population, List<Double> fitnessList, int popSize) {
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
