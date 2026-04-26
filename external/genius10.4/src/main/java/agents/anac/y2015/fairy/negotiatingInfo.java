package agents.anac.y2015.fairy;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;

public class negotiatingInfo {
	private AdditiveUtilitySpace utilitySpace; // ��p���
	private List<Issue> issues; // �_�_
	private ArrayList<Object> opponents; // ���g�ȊO�̌��Q���҂�sender
	private ArrayList<Bid> MyBidHistory = null; // ��ė���
	private HashMap<Object, ArrayList<Bid>> opponentsBidHistory = null; // ��ė���
	private HashMap<Object, Boolean> opponentsBool; // ��Ԃ��^�����ǂ���
	private HashMap<Object, Double> opponentsAverage; // ����
	private HashMap<Object, Double> opponentsVariance; // ���U
	private HashMap<Object, Double> opponentsSum; // �a
	private HashMap<Object, Double> opponentsPowSum; // ���a
	private HashMap<Object, Double> opponentsStandardDeviation; // �W���΍�
	private HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = null; // ���g�̌�p��Ԃɂ�����e�_�_�l�̑��Ό�p�l�s��i��`��p��ԗp�j
	private int round = 0; // �����̎�Ԑ�
	private int negotiatorNum = 3; // ���Ґ�
	private boolean isLinerUtilitySpace = true; // ��`��p��Ԃł��邩�ǂ���

	public negotiatingInfo(AdditiveUtilitySpace utilitySpace) {
		// ����
		this.utilitySpace = utilitySpace;
		issues = utilitySpace.getDomain().getIssues();
		opponents = new ArrayList<Object>();
		MyBidHistory = new ArrayList<Bid>();
		opponentsBidHistory = new HashMap<Object, ArrayList<Bid>>();
		opponentsBool = new HashMap<Object, Boolean>();
		opponentsAverage = new HashMap<Object, Double>();
		opponentsVariance = new HashMap<Object, Double>();
		opponentsSum = new HashMap<Object, Double>();
		opponentsPowSum = new HashMap<Object, Double>();
		opponentsStandardDeviation = new HashMap<Object, Double>();
		valueRelativeUtility = new HashMap<Issue, HashMap<Value, Double>>();

		try {
			initValueRelativeUtility();
		} catch (Exception e) {
			System.out.println("���Ό�p�s��̏���Ɏ��s���܂���");
			e.printStackTrace();
		}
	}

	public void initOpponent(Object sender) {
		initNegotiatingInfo(sender); // ����������
		opponents.add(sender); // ���Q���҂�sender��ǉ�
	}

	public void updateInfo(Object sender, Bid offeredBid) {
		try {
			updateNegotiatingInfo(sender, offeredBid);
		} // �����̍X�V
		catch (Exception e1) {
			System.out.println("�����̍X�V�Ɏ��s���܂���");
			e1.printStackTrace();
		}
	}

	private void initNegotiatingInfo(Object sender) {
		opponentsBidHistory.put(sender, new ArrayList<Bid>());
		opponentsAverage.put(sender, 0.0);
		opponentsVariance.put(sender, 0.0);
		opponentsSum.put(sender, 0.0);
		opponentsPowSum.put(sender, 0.0);
		opponentsStandardDeviation.put(sender, 0.0);
	}

	// ���Ό�p�s��̏���
	private void initValueRelativeUtility() throws Exception {
		ArrayList<Value> values = null;
		for (Issue issue : issues) {
			valueRelativeUtility.put(issue, new HashMap<Value, Double>()); // �_�_�s�̏���
			values = getValues(issue);
			for (Value value : values) {
				valueRelativeUtility.get(issue).put(value, 0.0);
			} // �_�_�s�̗v�f�̏���
		}
	}

	// ���Ό�p�s��̓��o
	public void setValueRelativeUtility(Bid maxBid) throws Exception {
		ArrayList<Value> values = null;
		Bid currentBid = null;
		for (Issue issue : issues) {
			currentBid = new Bid(maxBid);
			values = getValues(issue);
			for (Value value : values) {
				currentBid = currentBid.putValue(issue.getNumber(), value);
				valueRelativeUtility.get(issue).put(
						value,
						utilitySpace.getUtility(currentBid)
								- utilitySpace.getUtility(maxBid));
			}
		}
	}

	public void updateNegotiatingInfo(Object sender, Bid offeredBid)
			throws Exception {
		opponentsBidHistory.get(sender).add(offeredBid); // ��ė���

		double util = utilitySpace.getUtility(offeredBid);
		opponentsSum.put(sender, opponentsSum.get(sender) + util); // �a
		opponentsPowSum.put(sender,
				opponentsPowSum.get(sender) + Math.pow(util, 2)); // ���a

		int round_num = opponentsBidHistory.get(sender).size();
		opponentsAverage.put(sender, opponentsSum.get(sender) / round_num); // ����
		opponentsVariance.put(sender, (opponentsPowSum.get(sender) / round_num)
				- Math.pow(opponentsAverage.get(sender), 2)); // ���U

		if (opponentsVariance.get(sender) < 0) {
			opponentsVariance.put(sender, 0.0);
		}
		opponentsStandardDeviation.put(sender,
				Math.sqrt(opponentsVariance.get(sender))); // �W���΍�
	}

	// ���Ґ���Ԃ�
	public void updateOpponentsNum(int num) {
		negotiatorNum = num;
	}

	// ��`��p��ԂłȂ��ꍇ
	public void utilitySpaceTypeisNonLiner() {
		isLinerUtilitySpace = false;
	}

	// ���g�̒�ď��̍X�V
	public void updateMyBidHistory(Bid offerBid) {
		MyBidHistory.add(offerBid);
	}

	// �^�����ǂ���(�Q�b�g)
	public boolean getOpponentsBool(Object sender) {
		if (opponentsBool.isEmpty())
			return false;
		return opponentsBool.get(sender);
	}

	// �^�����ǂ���(�N���A)
	public void clearOpponentsBool() {
		opponentsBool.clear();
		return;
	}

	// �^�����ǂ���(�Z�b�g)
	public void setOpponentsBool(Object sender, boolean bool) {
		opponentsBool.put(sender, bool);
		return;
	}

	// ����
	public double getAverage(Object sender) {
		return opponentsAverage.get(sender);
	}

	// ���U
	public double getVariancer(Object sender) {
		return opponentsVariance.get(sender);
	}

	// �W���΍�
	public double getStandardDeviation(Object sender) {
		return opponentsStandardDeviation.get(sender);
	}

	// ����̒�ė����̗v�f����Ԃ�
	public int getPartnerBidNum(Object sender) {
		return opponentsBidHistory.get(sender).size();
	}

	// ���g�̃��E���h����Ԃ�
	public int getRound() {
		return round;
	}

	// ���Ґ���Ԃ�
	public int getNegotiatorNum() {
		return negotiatorNum;
	}

	// ���Ό�p�s���Ԃ�
	public HashMap<Issue, HashMap<Value, Double>> getValueRelativeUtility() {
		return valueRelativeUtility;
	}

	// ��`��p��Ԃł��邩�ǂ�����Ԃ�
	public boolean isLinerUtilitySpace() {
		return isLinerUtilitySpace;
	}

	// �_�_�ꗗ��Ԃ�
	public List<Issue> getIssues() {
		return issues;
	}

	// �_�_�ɂ������蓾��l�̈ꗗ��Ԃ�
	public ArrayList<Value> getValues(Issue issue) {
		ArrayList<Value> values = new ArrayList<Value>();
		switch (issue.getType()) {
		case DISCRETE:
			List<ValueDiscrete> valuesDis = ((IssueDiscrete) issue).getValues();
			for (Value value : valuesDis) {
				values.add(value);
			}
			break;
		case INTEGER:
			int min_value = ((IssueInteger) issue).getUpperBound();
			int max_value = ((IssueInteger) issue).getUpperBound();
			for (int j = min_value; j <= max_value; j++) {
				Object valueObject = new Integer(j);
				values.add((Value) valueObject);
			}
			break;
		default:
			try {
				throw new Exception("issue type " + issue.getType()
						+ " not supported by Atlas3");
			} catch (Exception e) {
				System.out.println("�_�_�̎�蓾��l�̎擾�Ɏ��s���܂���");
				e.printStackTrace();
			}
		}
		return values;
	}

	// ������̈ꗗ��Ԃ�
	public ArrayList<Object> getOpponents() {
		return opponents;
	}
}
