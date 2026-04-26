package agents.anac.y2015.fairy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.utility.AdditiveUtilitySpace;

public class bidSearch {
	private AdditiveUtilitySpace utilitySpace;
	private negotiatingInfo negotiatingInfo; // �����
	private Bid maxBid = null; // �ő��p�lBid

	// �T���̃p�����[�^
	private static int SA_ITERATION = 1;
	static double START_TEMPERATURE = 1.0; // �J�n���x
	static double END_TEMPERATURE = 0.0001; // �I�����x
	static double COOL = 0.999; // ��p�x
	static int STEP = 1;// �ύX���镝
	static int STEP_NUM = 1; // �ύX�����

	public bidSearch(AdditiveUtilitySpace utilitySpace,
			negotiatingInfo negotiatingInfo) throws Exception {
		this.utilitySpace = utilitySpace;
		this.negotiatingInfo = negotiatingInfo;
		initMaxBid(); // �ő��p�lBid�̏���T��
		negotiatingInfo.setValueRelativeUtility(maxBid); // ���Ό�p�l�𓱏o����
	}

	// �ő��p�lBid�̏���T��(�ŏ��͌�p��Ԃ̃^�C�v���s���ł��邽�߁CSA��p���ĒT������)
	private void initMaxBid() throws Exception {
		int tryNum = utilitySpace.getDomain().getIssues().size(); // ���s��
		maxBid = utilitySpace.getDomain().getRandomBid(null);
		for (int i = 0; i < tryNum; i++) {
			try {
				do {
					SimulatedAnnealingSearch(maxBid, 1.0);
				} while (utilitySpace.getUtility(maxBid) < utilitySpace
						.getReservationValue());
				if (utilitySpace.getUtility(maxBid) == 1.0) {
					break;
				}
			} catch (Exception e) {
				System.out.println("�ő��p�lBid�̏���T���Ɏ��s���܂���");
				e.printStackTrace();
			}
		}
	}

	// Bid��Ԃ�
	public Bid getBid(Bid baseBid, double threshold) {
		// Type:Real�ɑΉ��i�b��Łj
		for (Issue issue : negotiatingInfo.getIssues()) {
			switch (issue.getType()) {
			case REAL:
				try {
					return (getRandomBid(threshold));
				} catch (Exception e) {
					System.out.println("Bid�̃����_���T���Ɏ��s���܂���(Real)");
					e.printStackTrace();
				}
				break;
			default:
				break;
			}
		}

		// Type:Integer and Discrete
		try {
			Bid bid = getBidbyAppropriateSearch(baseBid, threshold); // 臒l�ȏ�̌�p�l�������ӈČ���T��
			if (utilitySpace.getUtility(bid) < threshold) {
				bid = new Bid(maxBid);
			} // �T���ɂ���ē���ꂽBid��threshold�����������ꍇ�C�ő��p�lBid����Ƃ���
			return bid;
		} catch (Exception e) {
			System.out.println("Bid�̒T���Ɏ��s���܂���");
			e.printStackTrace();
			return baseBid;
		}
	}

	// �����_���T��
	private Bid getRandomBid(double threshold) throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
																		// <issuenumber,chosen
																		// value
																		// string>
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();

		Bid bid = null;
		do {
			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					int optionIndex = randomnr.nextInt(lIssueDiscrete
							.getNumberOfValues());
					values.put(lIssue.getNumber(),
							lIssueDiscrete.getValue(optionIndex));
					break;
				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					int optionInd = randomnr.nextInt(lIssueReal
							.getNumberOfDiscretizationSteps() - 1);
					values.put(
							lIssueReal.getNumber(),
							new ValueReal(lIssueReal.getLowerBound()
									+ (lIssueReal.getUpperBound() - lIssueReal
											.getLowerBound())
									* (double) (optionInd)
									/ (double) (lIssueReal
											.getNumberOfDiscretizationSteps())));
					break;
				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					int optionIndex2 = lIssueInteger.getLowerBound()
							+ randomnr.nextInt(lIssueInteger.getUpperBound()
									- lIssueInteger.getLowerBound());
					values.put(lIssueInteger.getNumber(), new ValueInteger(
							optionIndex2));
					break;
				default:
					throw new Exception("issue type " + lIssue.getType()
							+ " not supported by Atlas3");
				}
			}
			bid = new Bid(utilitySpace.getDomain(), values);
		} while (utilitySpace.getUtility(bid) < threshold);

		return bid;
	}

	// Bid�̒T��
	private Bid getBidbyAppropriateSearch(Bid baseBid, double threshold) {
		Bid bid = new Bid(baseBid);
		try {
			// ��`��p��ԗp�̒T��
			if (negotiatingInfo.isLinerUtilitySpace()) {
				bid = relativeUtilitySearch(threshold);
				if (utilitySpace.getUtility(bid) < threshold) {
					negotiatingInfo.utilitySpaceTypeisNonLiner();
				} // �T���Ɏ��s�����ꍇ�C���`��p��ԗp�̒T���ɐ؂�ւ���
			}

			// ���`��p��ԗp�̒T��
			if (!negotiatingInfo.isLinerUtilitySpace()) {
				Bid currentBid = null;
				double currentBidUtil = 0;
				double min = 1.0;
				for (int i = 0; i < SA_ITERATION; i++) {
					currentBid = SimulatedAnnealingSearch(bid, threshold);
					currentBidUtil = utilitySpace.getUtility(currentBid);
					if (currentBidUtil <= min && currentBidUtil >= threshold) {
						bid = new Bid(currentBid);
						min = currentBidUtil;
					}
				}
			}
		} catch (Exception e) {
			System.out.println("SA�T���Ɏ��s���܂���");
			System.out.println("Problem with received bid(SA:last):"
					+ e.getMessage() + ". cancelling bidding");
		}
		return bid;
	}

	// ���Ό�p�l�Ɋ�Â��T��
	private Bid relativeUtilitySearch(double threshold) throws Exception {
		Bid bid = new Bid(maxBid);
		double d = threshold - 1.0; // �ő��p�l�Ƃ̍�
		double concessionSum = 0.0; // ���炵����p�l�̘a
		double relativeUtility = 0.0;
		HashMap<Issue, HashMap<Value, Double>> valueRelativeUtility = negotiatingInfo
				.getValueRelativeUtility();
		List<Issue> randomIssues = negotiatingInfo.getIssues();
		Collections.shuffle(randomIssues);
		ArrayList<Value> randomValues = null;
		for (Issue issue : randomIssues) {
			randomValues = negotiatingInfo.getValues(issue);
			Collections.shuffle(randomValues);
			for (Value value : randomValues) {
				relativeUtility = valueRelativeUtility.get(issue).get(value); // �ő��p�l����Ƃ������Ό�p�l
				if (d <= concessionSum + relativeUtility) {
					bid = bid.putValue(issue.getNumber(), value);
					concessionSum += relativeUtility;
					break;
				}
			}
		}
		return bid;
	}

	// SA
	private Bid SimulatedAnnealingSearch(Bid baseBid, double threshold)
			throws Exception {
		Bid currentBid = new Bid(baseBid); // ������̐���
		double currenBidUtil = utilitySpace.getUtility(baseBid);
		Bid nextBid = null; // �]��Bid
		double nextBidUtil = 0.0;
		ArrayList<Bid> targetBids = new ArrayList<Bid>(); // �œK��p�lBid��ArrayList
		double targetBidUtil = 0.0;
		double p; // �J�ڊm��
		Random randomnr = new Random(); // ����
		double currentTemperature = START_TEMPERATURE; // ���݂̉��x
		double newCost = 1.0;
		double currentCost = 1.0;
		List<Issue> issues = negotiatingInfo.getIssues();

		while (currentTemperature > END_TEMPERATURE) { // ���x���\��������܂Ń��[�v
			nextBid = new Bid(currentBid); // next_bid������
			for (int i = 0; i < STEP_NUM; i++) { // �ߖT��Bid���擾����
				int issueIndex = randomnr.nextInt(issues.size()); // �_�_�������_���Ɏw��
				Issue issue = issues.get(issueIndex); // �w�肵��index��issue
				ArrayList<Value> values = negotiatingInfo.getValues(issue);
				int valueIndex = randomnr.nextInt(values.size()); // ��蓾��l�͈̔͂Ń����_���Ɏw��
				nextBid = nextBid.putValue(issue.getNumber(),
						values.get(valueIndex));
				nextBidUtil = utilitySpace.getUtility(nextBid);
				if (maxBid == null
						|| nextBidUtil >= utilitySpace.getUtility(maxBid)) {
					maxBid = new Bid(nextBid);
				} // �ő��p�lBid�̍X�V
			}

			newCost = Math.abs(threshold - nextBidUtil);
			currentCost = Math.abs(threshold - currenBidUtil);
			p = Math.exp(-Math.abs(newCost - currentCost) / currentTemperature);
			if (newCost < currentCost || p > randomnr.nextDouble()) {
				currentBid = new Bid(nextBid); // Bid�̍X�V
				currenBidUtil = nextBidUtil;
			}

			// �X�V
			if (currenBidUtil >= threshold) {
				if (targetBids.size() == 0) {
					targetBids.add(new Bid(currentBid));
					targetBidUtil = utilitySpace.getUtility(currentBid);
				} else {
					if (currenBidUtil < targetBidUtil) {
						targetBids.clear(); // ����
						targetBids.add(new Bid(currentBid)); // �v�f��ǉ�
						targetBidUtil = utilitySpace.getUtility(currentBid);
					} else if (currenBidUtil == targetBidUtil) {
						targetBids.add(new Bid(currentBid)); // �v�f��ǉ�
					}
				}
			}
			currentTemperature = currentTemperature * COOL; // ���x��������
		}

		if (targetBids.size() == 0) {
			return new Bid(baseBid);
		} // ���E�l���傫�Ȍ�p�l������Bid��������Ȃ������Ƃ��́CbaseBid��Ԃ�
		else {
			return new Bid(targetBids.get(randomnr.nextInt(targetBids.size())));
		} // ��p�l�����E�l�t�߂ƂȂ�Bid��Ԃ�
	}
}