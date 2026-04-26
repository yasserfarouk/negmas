package list;

import static org.junit.Assert.assertEquals;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import genius.core.list.ImArrayList;
import genius.core.list.ImmutableList;
import genius.core.list.ListWithRemove;
import genius.core.list.PermutationsWithReturn;

public class ListWithRemoveTest {

	final static List<String> data = Arrays.asList(new String[] { "a", "b", "c", "d", "e" });

	@Test
	public void RemoveTest0() {
		ListWithRemove<String> remList = new ListWithRemove<>(new ImArrayList<>(data));
		assertEquals("a", remList.get(BigInteger.ZERO));
		remList = remList.remove(BigInteger.ZERO);
		assertEquals("b", remList.get(BigInteger.ZERO));
		assertEquals(BigInteger.valueOf(4), remList.size());
	}

	@Test
	public void RemoveTest1() {
		ListWithRemove<String> remList = new ListWithRemove<>(new ImArrayList<>(data));
		assertEquals("a", remList.get(BigInteger.ZERO));
		remList = remList.remove(BigInteger.ONE);
		assertEquals("a", remList.get(BigInteger.ZERO));
		assertEquals(BigInteger.valueOf(4), remList.size());
	}

	@Test
	public void RemoveTest1b() {
		ListWithRemove<String> remList = new ListWithRemove<>(new ImArrayList<>(data));
		assertEquals("b", remList.get(BigInteger.ONE));
		remList = remList.remove(BigInteger.ONE);
		assertEquals("c", remList.get(BigInteger.ONE));
		assertEquals(BigInteger.valueOf(4), remList.size());
	}

	@Test
	public void RemoveTestLarge() {
		PermutationsWithReturn<String> perm = new PermutationsWithReturn<>(new ImArrayList<>(data), 10);
		// assertEquals(BigInteger.valueOf(9765625), perm.size());
		ListWithRemove<ImmutableList<String>> permutation = new ListWithRemove<>(perm);
		System.out.println(permutation);
		permutation.remove(BigInteger.valueOf(838232));
		System.out.println(permutation);

	}

	@Test
	public void testRemoveListMultipleTimes() {

		for (int n = 0; n < 20; n++) {
			testRemoveRandomTillEmpty();
		}
	}

	/**
	 * remove random data items till lists are empty
	 */
	@Test
	public void testRemoveRandomTillEmpty() {
		ImmutableList<String> list = new ImArrayList<>(data);
		ListWithRemove<String> remlist = new ListWithRemove<>(list);
		ArrayList<String> copy = new ArrayList<>(data);

		while (!copy.isEmpty()) {
			Random r = new Random();

			int n = r.nextInt(copy.size());
			remlist = remlist.remove(BigInteger.valueOf(n));
			copy.remove(n);

			checkListsEqual(remlist, copy);
		}

	}

	@Test
	public void testImmutability() {
		List<String> copy = new ArrayList<>(data);

		ImmutableList<String> list = new ImArrayList<>(copy);
		copy.remove(1);
		checkListsEqual(list, data);
	}

	private void checkListsEqual(ImmutableList<String> list1, List<String> list2) {
		assertEquals(list1.size().intValue(), list2.size());
		for (int n = 0; n < list2.size(); n++) {
			assertEquals(list1.get(BigInteger.valueOf(n)), list2.get(n));
		}

	}
}
