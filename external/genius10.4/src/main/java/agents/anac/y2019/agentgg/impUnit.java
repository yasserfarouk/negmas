package agents.anac.y2019.agentgg;

import genius.core.issue.Value;

import java.util.Comparator;

//importance单元
public class impUnit {
    public Value valueOfIssue;
    public int weightSum = 0;
    public int count = 0;
    public double meanWeightSum = 0.0f;

    public impUnit(Value value) {
        this.valueOfIssue = value;
    }

    public String toString() {
        return String.format("%s %f", valueOfIssue, meanWeightSum);
    }


    //重写comparator接口
    static class meanWeightSumComparator implements Comparator<impUnit> {
        public int compare(impUnit o1, impUnit o2) {// 实现接口中的方法
            if (o1.meanWeightSum < o2.meanWeightSum) {
                return 1;
            } else if (o1.meanWeightSum > o2.meanWeightSum) {
                return -1;
            }
            return 0;
        }
    }

}
