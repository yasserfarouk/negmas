package list;

import static org.junit.Assert.assertEquals;

import java.math.BigInteger;
import java.util.Arrays;

import org.junit.Test;

import genius.core.list.ImArrayList;
import genius.core.list.PermutationsOrderedWithoutReturn;

public class PermutationOrderedWithoutReturnTest {
	@Test
	public void test1() {
		ImArrayList<String> source = new ImArrayList<String>(Arrays.asList(new String[] { "a", "b", "c" }));
		PermutationsOrderedWithoutReturn<String> p = new PermutationsOrderedWithoutReturn<String>(source, 1);
		assertEquals(BigInteger.valueOf(3), p.size());
		System.out.println("data=" + p + "\n");
		assertEquals("[[c],[b],[a]]", p.toString());
	}

	@Test
	public void test2() {
		ImArrayList<String> source = new ImArrayList<String>(Arrays.asList(new String[] { "a", "b", "c", "d" }));
		PermutationsOrderedWithoutReturn<String> p = new PermutationsOrderedWithoutReturn<String>(source, 2);
		assertEquals(BigInteger.valueOf(6), p.size());
		System.out.println("data=" + p + "\n");
		assertEquals("[[c, d],[b, d],[b, c],[a, d],[a, c],[a, b]]", p.toString());
	}

}
