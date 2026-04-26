package agents.anac.y2019.agentgg;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.uncertainty.UserModel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ImpMap extends HashMap<Issue, List<impUnit>> {
    //importance map
    public ImpMap(UserModel userModel) {
        super();
        //遍历userModel中的issue，创建空importance表格
        for (Issue issue : userModel.getDomain().getIssues()) {
            IssueDiscrete temp = (IssueDiscrete) issue;
            List<impUnit> issueImpUnit = new ArrayList<>();
            int numberInIssue = temp.getNumberOfValues();
            for (int i = 0; i < numberInIssue; i++) {
                issueImpUnit.add(new impUnit(temp.getValue(i)));
            }
            this.put(issue, issueImpUnit);
        }
    }

    // 更新对手map
    public void opponent_update(Bid receivedOfferBid) {
        for (Issue issue : receivedOfferBid.getIssues()) {
            int no = issue.getNumber();
            List<impUnit> currentIssueList = this.get(issue);
            for (impUnit currentUnit : currentIssueList) {
                if (currentUnit.valueOfIssue.toString().equals(receivedOfferBid.getValue(no).toString())) {
                    currentUnit.meanWeightSum += 1;
                    break;
                }
            }
        }
        for (List<impUnit> impUnitList : this.values()) {
            impUnitList.sort(new impUnit.meanWeightSumComparator());
        }
    }

    // 更新自己的importance map
    public void self_update(UserModel userModel) {
        //遍历已知bidOrder，更新importance表格中的“权和”、“次数”
        int currentWeight = 0;
        for (Bid bid : userModel.getBidRanking().getBidOrder()) {
            currentWeight += 1;
            List<Issue> issueList = bid.getIssues();
            for (Issue issue : issueList) {
                int no = issue.getNumber();
                List<impUnit> currentIssueList = this.get(issue);
                for (impUnit currentUnit : currentIssueList) {
                    if (currentUnit.valueOfIssue.toString().equals(bid.getValue(no).toString())) {
                        currentUnit.weightSum += currentWeight;
                        currentUnit.count += 1;
                        break;
                    }
                }
            }
        }
        // 计算权重
        for (List<impUnit> impUnitList : this.values()) {
            for (impUnit currentUnit : impUnitList) {
                if (currentUnit.count == 0) {
                    currentUnit.meanWeightSum = 0.0;
                } else {
                    currentUnit.meanWeightSum = (double) currentUnit.weightSum / (double) currentUnit.count;
                }
            }
        }
        // 排序
        for (List<impUnit> impUnitList : this.values()) {
            impUnitList.sort(new impUnit.meanWeightSumComparator());
        }
        // 找到最小值
        double minMeanWeightSum = Double.POSITIVE_INFINITY;
        for (Map.Entry<Issue, List<impUnit>> entry : this.entrySet()) {
            double tempMeanWeightSum = entry.getValue().get(entry.getValue().size() - 1).meanWeightSum;
            if (tempMeanWeightSum < minMeanWeightSum) {
                minMeanWeightSum = tempMeanWeightSum;
            }
        }
        // 所有值都减去最小值
        for (List<impUnit> impUnitList : this.values()) {
            for (impUnit currentUnit : impUnitList) {
                currentUnit.meanWeightSum -= minMeanWeightSum;
            }
        }
    }

    //计算某个bid对应的importance值
    public double getImportance(Bid bid) {
        double bidImportance = 0.0;
        for (Issue issue : bid.getIssues()) {
            Value value = bid.getValue(issue.getNumber());
            double valueImportance = 0.0;
            for (impUnit i : this.get(issue)) {
                if (i.valueOfIssue.equals(value)) {
                    valueImportance = i.meanWeightSum;
                    break;
                }
            }
            bidImportance += valueImportance;
        }
        return bidImportance;
    }
}

