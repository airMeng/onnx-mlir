/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- ArgMin.cpp - Lowering ArgMin Op -------------------===//
//
// Copyright 2021-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX ArgMin Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {
struct ONNXArgMinOpLowering : public ConversionPattern {
  ONNXArgMinOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXArgMinOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    auto loc = op->getLoc();
    ONNXArgMinOpAdaptor operandAdaptor(operands);
    ONNXArgMinOp argMinOp = llvm::cast<ONNXArgMinOp>(op);

    // shape helper
    ONNXArgMinOpShapeHelper shapeHelper(&argMinOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);

    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    (void)shapecomputed;
    assert(!failed(shapecomputed) && "expected to succeed");

    // Convert the reduced output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType reducedMemRefType = convertedType.cast<MemRefType>();
    Type reducedElementType = reducedMemRefType.getElementType();
    int64_t reducedRank = reducedMemRefType.getRank();

    // data input
    auto data = operandAdaptor.data();
    auto dataType = data.getType().cast<MemRefType>();
    int64_t dataRank = dataType.getRank();

    // axis & keepdims attribute
    int64_t axis = argMinOp.axis();
    assert(axis >= -dataRank && axis <= dataRank - 1);
    axis = axis >= 0 ? axis : (dataRank + axis);

    int64_t keepdims = argMinOp.keepdims();
    bool isKeepdims = (keepdims == 1) ? true : false;

    // Get type information
    llvm::SmallVector<int64_t, 1> axes;
    axes.push_back(axis);
    std::map<int64_t, int64_t> outInDimMap =
        getReductionMapping(dataType, llvm::makeArrayRef(axes), isKeepdims);

    // Insert alloc and dealloc
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, reducedMemRefType, loc, shapeHelper.dimsForOutput());

    // Constant Value
    MathBuilder createMath(rewriter, loc);
    Value minusOne = createMath.constant(reducedElementType, -1);
    Value zero = createMath.constant(reducedElementType, 0);
    auto zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    KrnlBuilder createKrnl(rewriter, loc);

    // 1. Krnl loops to initialize the result.
    ValueRange initLoopDef = createKrnl.defineLoops(reducedRank);
    SmallVector<IndexExpr, 4> initLbs(reducedRank, LiteralIndexExpr(0));
    createKrnl.iterateIE(initLoopDef, initLoopDef, initLbs,
        shapeHelper.dimsForOutput(0),
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          createKrnl.store(minusOne, alloc, loopInd);
        });

    // 2. Krnl loop to calculate argmin.
    MultiDialectBuilder<KrnlBuilder, MathBuilder> create(rewriter, loc);
    ValueRange calcLoopDef = createKrnl.defineLoops(dataRank);
    SmallVector<IndexExpr, 4> lbs(dataRank, LiteralIndexExpr(0));
    MemRefBoundsIndexCapture dataBounds(data);
    SmallVector<IndexExpr, 4> ubs;
    dataBounds.getDimList(ubs);
    createKrnl.iterateIE(calcLoopDef, calcLoopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Handle the operation:
          SmallVector<Value, 4> inLoopIVs, outLoopIVs, minLoopIVs;

          for (int i = 0; i < dataRank; ++i)
            inLoopIVs.push_back(loopInd[i]);

          for (int i = 0; i < reducedRank; ++i) {
            if (outInDimMap.find(i) != outInDimMap.end())
              outLoopIVs.push_back(inLoopIVs[outInDimMap[i]]);
            else
              outLoopIVs.push_back(zeroIndex);
          }

          Value next = createKrnl.load(data, inLoopIVs);
          Value idx = createKrnl.load(alloc, outLoopIVs);

          // if index is less than 0, we should set 0 as initial position
          Value lessThanZero = create.math.slt(idx, zero);
          idx = create.math.select(lessThanZero, zero, idx);

          // induction variables of current min value
          for (int i = 0; i < dataRank; ++i) {
            if (i != axis)
              minLoopIVs.push_back(loopInd[i]);
            else
              minLoopIVs.push_back(rewriter.create<arith::IndexCastOp>(
                  loc, rewriter.getIndexType(), idx));
          }
          Value minVal = createKrnl.load(data, minLoopIVs);

          // if next value is larger than current min value, update index
          Value greaterThanMin = create.math.slt(next, minVal);
          Value pos = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getIntegerType(64), inLoopIVs[axis]);
          idx = create.math.select(greaterThanMin, pos, idx);
          createKrnl.store(idx, alloc, outLoopIVs);
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXArgMinOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXArgMinOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
