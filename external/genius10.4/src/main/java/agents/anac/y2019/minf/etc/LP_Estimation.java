package agents.anac.y2019.minf.etc;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.uncertainty.AdditiveUtilitySpaceFactory;
import genius.core.uncertainty.BidRanking;
import genius.core.uncertainty.OutcomeComparison;

import agents.org.apache.commons.math.optimization.linear.*;
import agents.org.apache.commons.math.optimization.RealPointValuePair;
import agents.org.apache.commons.math.optimization.GoalType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LP_Estimation {
    private AdditiveUtilitySpaceFactory additiveUtilitySpaceFactory;
    private BidRanking bidRanking;
    private List<IssueDiscrete> issues;

    public LP_Estimation(){
    }

    public LP_Estimation(Domain d, BidRanking r){
        additiveUtilitySpaceFactory = new AdditiveUtilitySpaceFactory(d);
        issues = additiveUtilitySpaceFactory.getIssues();
        bidRanking = r;
    }

    public AdditiveUtilitySpaceFactory Estimation()
            throws Exception{
        int issue_num = issues.size();
        int[] var_num = new int[issue_num];
        int[] cum_sum = new int[issue_num+1];
        int num_vars = 0;	// phyの数（求めたいW*V）
        int num_comp = bidRanking.getSize() - 1;
        double high = bidRanking.getHighUtility();
        double low = bidRanking.getLowUtility();

        for (IssueDiscrete i : issues){
            int itr = i.getNumber()-1;
            var_num[itr] = i.getNumberOfValues();
            cum_sum[itr] = num_vars;
            num_vars += i.getNumberOfValues();
        }
        cum_sum[cum_sum.length-1] = num_vars;

        // 制約式の数がvars+comp*2+2，変数の数がvars+comp
        double[][] cons_A = new double[num_vars + num_comp * 2 + 2][num_vars + num_comp];
        double[] cons_b = new double[num_vars + num_comp * 2 + 2];
        double[] obj_c = new double[num_vars + num_comp];
        Arrays.fill(cons_b, 0.0D);
        Arrays.fill(obj_c, 0.0D);
        for (int i = 0; i < num_comp; i++){ obj_c[num_vars + i] = 1.0D; }

        int pos = 0;

        for (OutcomeComparison c : bidRanking.getPairwiseComparisons()){
            Bid bid_low = c.getBid1();
            Bid bid_high = c.getBid2();

            for (IssueDiscrete i : issues){
                int itr = i.getNumber()-1;
                if (!bid_low.getValue(i).equals(bid_high.getValue(i))) {
                    cons_A[pos][cum_sum[itr] + i.getValueIndex(bid_low.getValue(i).toString())] = -1.0D;
                    cons_A[pos][cum_sum[itr] + i.getValueIndex(bid_high.getValue(i).toString())] = 1.0D;
                }
            }
            cons_A[pos][num_vars + pos] = 1.0D;
            pos++;
        }

        for (int i = 0; i < num_vars + num_comp; i++){ cons_A[pos++][i] = 1.0D; }

        for (IssueDiscrete i : issues){
            int itr = i.getNumber()-1;
            cons_A[pos][cum_sum[itr] + i.getValueIndex(bidRanking.getMaximalBid().getValue(i).toString())] = 1.0D;
            cons_A[pos+1][cum_sum[itr] + i.getValueIndex(bidRanking.getMinimalBid().getValue(i).toString())] = 1.0D;
        }
        cons_b[pos] = high;
        cons_b[pos+1] = low;

        LinearObjectiveFunction lof = new LinearObjectiveFunction(obj_c, 0.0D);
        List<LinearConstraint> lc = new ArrayList<>();
        for (int i = 0; i < cons_A.length-2; i++) {
            lc.add(new LinearConstraint(cons_A[i], Relationship.GEQ, cons_b[i]));
        }
        lc.add(new LinearConstraint(cons_A[cons_A.length-2], Relationship.EQ, cons_b[cons_b.length-2]));
        lc.add(new LinearConstraint(cons_A[cons_A.length-1], Relationship.EQ, cons_b[cons_b.length-1]));

        SimplexSolver ss = new SimplexSolver();
        ss.setMaxIterations(2147483647);

        RealPointValuePair pvp = ss.optimize(lof, lc, GoalType.MINIMIZE, false);

        double[] optimal = Arrays.copyOfRange(pvp.getPoint(), 0, num_vars);
        for (int i = 0; i < optimal.length; i++){ optimal[i] = Math.max(0.0D, optimal[i]); }

        for (IssueDiscrete i : issues) {
            int itr = i.getNumber()-1;
            double max = 0.0;
            double[] tmp = Arrays.copyOfRange(optimal, cum_sum[itr], cum_sum[itr+1]);

            for (double d : tmp) { max = Math.max(d, max); }

            additiveUtilitySpaceFactory.setWeight(i, max);

            for (ValueDiscrete v : i.getValues()) {
                additiveUtilitySpaceFactory.setUtility(i, v, tmp[i.getValueIndex(v)]);
            }
        }

        additiveUtilitySpaceFactory.normalizeWeights();

        return additiveUtilitySpaceFactory;
    }
}
